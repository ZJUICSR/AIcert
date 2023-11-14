# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights   rved.
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

from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn as nn

from function.formal_verify.auto_LiRPA_verifiy.language.Transformer.modeling import BertForSequenceClassification, BertConfig
from function.formal_verify.auto_LiRPA_verifiy.language.Transformer.utils import convert_examples_to_features
from function.formal_verify.auto_LiRPA_verifiy.language.language_utils import build_vocab

from function.formal_verify.auto_LiRPA_verifiy.language.data_utils import get_sst_data

PRE_TRAIN_MODEL = os.path.join("model/auto_LiRPA_model",
                               'ckpt_transformer')


class Transformer(nn.Module):
    def __init__(self,
                 embedding_size=64,
                 max_sent_length=32,
                 min_word_freq=2,
                 device='cpu',
                 lr=1e-4,
                 dir='model',
                 load=None,
                 data_train=None,
                 hidden_size=64,
                 num_classes=2,
                 dropout=0.1,
                 num_attention_heads=4,
                 intermediate_size=128,
                 num_layers=1,
                 drop_unk='store_true',
                 hidden_act='relu',
                 layer_norm='no_var'):
        super().__init__()
        self.max_seq_length = max_sent_length
        self.drop_unk = drop_unk
        self.num_labels = num_classes
        self.label_list = range(num_classes)
        self.device = device
        self.lr = lr

        self.dir = dir
        self.vocab = build_vocab(data_train, min_word_freq)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = 0
        config = BertConfig(len(self.vocab))
        config.num_hidden_layers = num_layers
        config.embedding_size = embedding_size
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.hidden_act = hidden_act
        config.num_attention_heads = num_attention_heads
        config.layer_norm = layer_norm
        config.hidden_dropout_prob = dropout
        self.model = BertForSequenceClassification(
            config, self.num_labels, vocab=self.vocab).to(self.device)
        print("Model initialized")
        if load:
            checkpoint = torch.load(load, map_location=torch.device(self.device))
            epoch = checkpoint['epoch']
            self.model.embeddings.load_state_dict(checkpoint['state_dict_embeddings'])
            self.model.model_from_embeddings.load_state_dict(checkpoint['state_dict_model_from_embeddings'])
            print('Checkpoint loaded: {}'.format(load))

        self.model_from_embeddings = self.model.model_from_embeddings
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.model_from_embeddings.device = self.device

    def save(self, epoch):
        self.model.model_from_embeddings = self.model_from_embeddings
        path = os.path.join(self.dir, "ckpt_{}".format(epoch))
        torch.save({ 
            'state_dict_embeddings': self.model.embeddings.state_dict(), 
            'state_dict_model_from_embeddings': self.model.model_from_embeddings.state_dict(), 
            'epoch': epoch
        }, path)
        print("Model saved to {}".format(path))
        
    def build_optimizer(self):
        # update the original model with the converted model
        self.model.model_from_embeddings = self.model_from_embeddings
        param_group = [
            {"params": [p[1] for p in self.model.named_parameters()], "weight_decay": 0.},
        ]    
        return torch.optim.Adam(param_group, lr=self.lr)

    def train(self):
        self.model.train()
        self.model_from_embeddings.train()

    def eval(self):
        self.model.eval() 
        self.model_from_embeddings.eval()

    def get_input(self, batch):
        features = convert_examples_to_features(
            batch, self.label_list, self.max_seq_length, self.vocab, drop_unk=self.drop_unk)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(self.device)       
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)
        tokens = [f.tokens for f in features]

        embeddings, extended_attention_mask = \
            self.model(input_ids, segment_ids, input_mask, embed_only=True)

        return embeddings, extended_attention_mask, tokens, label_ids

    def forward(self, batch):
        embeddings, extended_attention_mask, tokens, label_ids = self.get_input(batch)
        logits = self.model_from_embeddings(embeddings, extended_attention_mask)        
        preds = torch.argmax(logits, dim=1)
        return preds


def get_transformer_model(load=PRE_TRAIN_MODEL, device='cpu'):
    print(f'get_transformer_model')
    _, data_train = get_sst_data()
    model = Transformer(load=load, data_train=data_train, device=device)
    return model


if __name__ == '__main__':
    pass

