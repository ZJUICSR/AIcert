# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forward step utilities."""

from collections.abc import Iterable
from functools import wraps

import torch

from megatron.training import get_args
from megatron.core import mpu, ModelParallelConfig, InferenceParams
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.inference.text_generation.forward_step import _get_recv_buffer_dtype


def inference_forward_step_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        if not args.use_kv_cache:
            self.inference_params = None

    return wrapper


def _forward_step_helper(self, tokens, position_ids, attention_mask, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    config = ModelParallelConfig
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage
    # Here use megatron/p2p do substitution which megatron want to do.
    recv_buffer = recv_buffer if mpu.is_pipeline_first_stage() else recv_buffer.shape
    input_tensor = p2p_communication.recv_forward(recv_buffer, config)

    # Forward pass through the model.
    self.model.set_input_tensor(input_tensor)
    output_tensor = self._forward(tokens, position_ids, attention_mask)

    # Send to the next stage.
    # Here use megatron/p2p do substitution which megatron want to do.
    p2p_communication.send_forward(output_tensor, config)

    return output_tensor


def _no_pipelining_forward_step_wrapper(_no_pipelining_forward_step):
    @wraps(_no_pipelining_forward_step)
    def wrapper(self, tokens, position_ids, attention_mask, recv_buffer=None):
        """If recv_buffer is none, we will allocate one on the fly."""
        # Run a simple forward pass.
        args = get_args()
        output_tensor = self._forward_step_helper(tokens, position_ids,
                                            attention_mask,
                                            recv_buffer=recv_buffer)
        # Update the sequence length offset.
        if self.inference_params:
            self.inference_params.sequence_len_offset += tokens.size(1)

        logits = None
        if mpu.is_pipeline_last_stage():
            if args.sequence_parallel:
                logits = gather_from_tensor_model_parallel_region(output_tensor)
            else:
                logits = output_tensor
        return logits
    return wrapper


def _with_pipelining_forward_step_wrapper(_with_pipelining_forward_step):
    @wraps(_with_pipelining_forward_step)
    def wrapper(self, tokens, position_ids, attention_mask, micro_batch_size):
        """No interleaving is supported."""
        args = get_args()
        sequence_length = tokens.size(1)
        batch_size = tokens.size(0)

        # Divide the batch dimension into micro batches.
        num_micro_batches, last_chunk = divmod(batch_size,
                                            micro_batch_size)
        if last_chunk > 0:
            num_micro_batches += 1

        # Preallocate memory for output logits.
        logits = None
        if mpu.is_pipeline_last_stage():
            args = get_args()
            if getattr(args, "task", False) and args.task[0] == 'needlebench':
                logits = torch.empty(
                    (batch_size, 100, args.padded_vocab_size),
                    dtype=torch.float32, device=torch.cuda.current_device())
            else:
                logits = torch.empty(
                    (batch_size, sequence_length, args.padded_vocab_size),
                    dtype=torch.float32, device=torch.cuda.current_device())


        # Preallocate recv buffer.
        recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

        for micro_batch_index in range(num_micro_batches):
            # Slice among the batch dimenion.
            start = micro_batch_index * micro_batch_size
            end = min(start + micro_batch_size, batch_size)
            this_micro_batch_size = end - start
            tokens2use = tokens[start:end, ...]
            position_ids2use = position_ids[start:end, ...]

            # Run a simple forward pass.
            if this_micro_batch_size != micro_batch_size:
                recv_buffer = None
            output = self._forward_step_helper(tokens2use, position_ids2use,
                                        attention_mask,
                                        recv_buffer=recv_buffer)

            if self.inference_params:
                # Adjust the batch size offset to account for the micro-batch.
                self.inference_params.batch_size_offset += this_micro_batch_size

            # Copy logits.
            if mpu.is_pipeline_last_stage():
                # Here for multi batches generation.
                if args.sequence_parallel:
                    output = gather_from_tensor_model_parallel_region(output)

                logits[start:end, ...] = output

        if self.inference_params:
            # Once we are done with all the micro-batches, we can
            # adjust the sequence length offset.
            self.inference_params.sequence_len_offset += sequence_length
            # and reset the batch size offset
            self.inference_params.batch_size_offset = 0

        return logits
    return wrapper


def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    if args.sequence_parallel:
        sequence_length = sequence_length // mpu.get_tensor_model_parallel_world_size()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())

