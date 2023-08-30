"""
These functions are used to parallelly compute reachable sets for the reachability analysis

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import numpy as np
from collections import deque


class Worker:
    """
    A class for the worker in the parallel computation

        Attributes:
            dnn (ffnn or cnn): A network model
            private_deque (deque): Deque to store local state tuples
            worker_id (int): Identity number of the worker
            shared_state (SharedState): Object for the communication between workers
            output_len (int): Length of the shared queue
            inital_num (int): Length threshold to start stealing work from the local deque
            inital_layer (int): Layer threshold to start stealing work from the local deque

        Methods:
            inital_assgin():
                Assign states to this local worker at the beginning of the computation
            main_func(indx, shared_state):
                Create and start parallel computation in this worker
            steal_from_this_worker():
                Steal states in the local deque of this worker
            asssign_to_this_worker():
                Assign shared states to this local worker
            collect_results(s):
                Conduct the function in terms of the configuration
            state_spawn_depth_first(tuple_state):
                Depth-first computation of states
            state_spawn_breath_first(tuple_state):
                Breath-first computation of states

    """
    def __init__(self, dnn, output_len=np.infty):
        self.dnn = dnn
        self.private_deque = deque()
        self.worker_id = None
        self.shared_state = None
        self.output_len = output_len
        self.inital_num = 500
        self.inital_layer = 2


    def inital_assgin(self):
        """
        Assign states to this local worker at the beginning of the computation
        """
        while True:
            try:
                one_work = self.shared_state.shared_queue.get_nowait()
                self.private_deque.append(one_work)
                with self.shared_state.shared_queue_len.get_lock():
                    self.shared_state.shared_queue_len.value = 0
            except:
                break

        with self.shared_state.initial_completed_workers.get_lock():
            self.shared_state.initial_completed_workers.value += 1

        # Wait until all workers are assigned with states
        if self.shared_state.initial_completed_workers.value == self.shared_state.num_workers:
            self.shared_state.initial_steal_assign.set()
        else:
            self.shared_state.initial_steal_assign.wait()


    def main_func(self, indx, shared_state):
        """
        Create and start parallel computation in this worker

        Parameters:
            indx (int): Index of this worker
            shared_state (SharedState): Object for the communication between workers
        """

        self.worker_id = indx
        self.shared_state = shared_state
        self.inital_assgin()

        # Initial breath-first computation to spawn many state tuples
        while self.private_deque:
            tuple_state = self.private_deque.popleft()
            self.state_spawn_breath_first(tuple_state)
            if len(self.private_deque) >= self.inital_num or tuple_state[1]==self.inital_layer:
                self.shared_state.steal_assign_ready.set()
                self.shared_state.workers_valid_status[self.worker_id] = 0
                self.shared_state.work_steal_ready.wait()
                self.steal_from_this_worker()
                break

        # Depth-first computation for output states of the network model
        while not (self.shared_state.work_interrupted.is_set()):
            while self.private_deque:
                tuple_state = self.private_deque.popleft()
                self.state_spawn_depth_first(tuple_state)
                #  Steal from this worker if conditions are satisfied
                if self.shared_state.workers_valid_status[self.worker_id] == 1 and self.shared_state.work_steal_ready.is_set():
                    self.steal_from_this_worker()
                if self.shared_state.work_interrupted.is_set():
                    break

            # Terminate this worker if the computation is done
            if self.shared_state.work_done.is_set():
                self.shared_state.workers_valid_status[self.worker_id] = 0
                self.shared_state.workers_idle_status[self.worker_id] = 1
                break

            with self.shared_state.workers_valid_status.get_lock():
                self.shared_state.workers_valid_status[self.worker_id] = 0
                # Prepare for work assigning
                if self.shared_state.work_steal_ready.is_set() and sum(self.shared_state.workers_valid_status) == 0:
                    self.shared_state.work_steal_ready.clear()

                    with self.shared_state.works_to_assign_per_worker.get_lock():
                        self.shared_state.works_to_assign_per_worker.value = \
                            np.ceil(self.shared_state.shared_queue_len.value / sum(self.shared_state.workers_to_assign)).astype(np.int64)
                    # Start work assigning for idle workers
                    self.shared_state.work_assign_ready.set()

            with self.shared_state.workers_idle_status.get_lock():
                self.shared_state.workers_idle_status[self.worker_id] = 1
                if sum(self.shared_state.workers_idle_status) == self.shared_state.num_workers:
                    if self.shared_state.shared_queue_len.value == 0:
                        self.shared_state.work_done.set()
                        self.shared_state.steal_assign_ready.set()
                        self.shared_state.work_steal_ready.clear()
                    self.shared_state.work_assign_ready.set()

            # This worker is waiting to enter the steal & assign mode
            self.shared_state.steal_assign_ready.wait()
            if self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set():
                self.shared_state.work_steal_ready.clear()
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_assign_ready.set()
                break

            with self.shared_state.workers_to_assign.get_lock():
                self.shared_state.workers_to_assign[self.worker_id] = 1
                with self.shared_state.workers_idle_status.get_lock():
                    # All idle workers enter the steal & assign mode
                    if sum(self.shared_state.workers_to_assign) == sum(self.shared_state.workers_idle_status):
                        self.shared_state.steal_assign_ready.clear() # Stop other workers' entry
                        self.shared_state.compute_steal_rate() # Compute the rate for the work stealing
                        with self.shared_state.num_empty_assign.get_lock():
                            self.shared_state.num_empty_assign.value = 0

                        # All the workers being idle indicates the computation is done
                        if (sum(self.shared_state.workers_to_assign) == self.shared_state.num_workers):
                            self.shared_state.work_done.set()
                            self.shared_state.steal_assign_ready.set()
                            assert self.shared_state.shared_queue_len.value==0
                            self.shared_state.work_steal_ready.clear()
                            self.shared_state.work_assign_ready.set()
                            break

                        # Start the work stealing from non-idle workers
                        self.shared_state.work_steal_ready.set()

            if self.shared_state.work_done.is_set() or self.shared_state.work_interrupted.is_set():
                self.shared_state.work_steal_ready.clear()
                self.shared_state.steal_assign_ready.set()
                self.shared_state.work_assign_ready.set()
                break

            # Wait for the work-assigning being ready
            self.shared_state.work_assign_ready.wait()
            self.shared_state.work_steal_ready.clear()
            self.asssign_to_this_worker()

        # Handle leftover states in the shared_queue
        if self.shared_state.work_done.is_set() and (not self.shared_state.work_interrupted.is_set()):
            with self.shared_state.num_workers_done.get_lock():
                self.shared_state.num_workers_done.value += 1
                if self.shared_state.num_workers_done.value == self.shared_state.num_workers:
                    self.shared_state.all_workers_done.set()
            self.shared_state.all_workers_done.wait()
            while True:
                with self.shared_state.shared_queue_len.get_lock():
                    if self.shared_state.shared_queue_len.value == 0:
                        break
                    one_state = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_state)
                    self.shared_state.shared_queue_len.value -= 1
                while self.private_deque:
                    tuple_state = self.private_deque.popleft()
                    self.state_spawn_depth_first(tuple_state)

            assert self.shared_state.shared_queue_len.value == 0
            if len(self.private_deque) != 0:
                print('error')


    def steal_from_this_worker(self):
        """
        Steal states in the local deque of this worker
        """
        steal_rate = self.shared_state.work_steal_rate.value
        assert steal_rate != 0

        num_stolen_works = 0
        if len(self.private_deque) >= 5: # Threshold for the stealing
            with self.shared_state.shared_queue_len.get_lock():
                num_stolen_works = np.floor(len(self.private_deque) * steal_rate).astype(np.int64)
                if num_stolen_works == 0:
                    num_stolen_works = 1

                for n in range(num_stolen_works):
                    # Steal states that generated earlier
                    stolen_work = self.private_deque.popleft()
                    self.shared_state.shared_queue.put(stolen_work)
                    self.shared_state.shared_queue_len.value += 1
                    with self.shared_state.stolen_works.get_lock():
                        self.shared_state.stolen_works.value += 1

        with self.shared_state.workers_valid_status.get_lock():
            # Indicate that this worker has been stolen
            self.shared_state.workers_valid_status[self.worker_id] = 0
            # If all the non-idle workers have been stolen
            if sum(self.shared_state.workers_valid_status) == 0:
                self.shared_state.work_steal_ready.clear()
                assert self.shared_state.stolen_works.value==self.shared_state.shared_queue_len.value
                with self.shared_state.works_to_assign_per_worker.get_lock():
                    self.shared_state.works_to_assign_per_worker.value = np.ceil(
                        self.shared_state.shared_queue_len.value / sum(self.shared_state.workers_to_assign)).astype(
                        np.int64)
                # Start assigning states to the idle workers
                self.shared_state.work_assign_ready.set()


    def asssign_to_this_worker(self):
        """
        Assign shared states to this local worker
        """
        with self.shared_state.assigned_works.get_lock():
            num = 0
            # Assign states to this worker
            while num < self.shared_state.works_to_assign_per_worker.value:
                with self.shared_state.shared_queue_len.get_lock():
                    if self.shared_state.shared_queue_len.value == 0:
                        break
                    one_work = self.shared_state.shared_queue.get()
                    self.private_deque.append(one_work)
                    self.shared_state.assigned_works.value += 1
                    self.shared_state.shared_queue_len.value -= 1
                num += 1

            if num == 0: # If this worker is not assigned
                with self.shared_state.num_empty_assign.get_lock():
                    self.shared_state.num_empty_assign.value += 1
            else:
                with self.shared_state.workers_idle_status.get_lock():
                    self.shared_state.workers_idle_status[self.worker_id] = 0

            with self.shared_state.workers_assigned.get_lock():
                self.shared_state.workers_assigned[self.worker_id] = 1
                if not (sum(self.shared_state.workers_assigned) <= sum(self.shared_state.workers_to_assign)):
                    assert sum(self.shared_state.workers_assigned) <= sum(self.shared_state.workers_to_assign)

                if sum(self.shared_state.workers_assigned) == sum(self.shared_state.workers_to_assign): # Work assigning is done
                    # Total assigned states can be less than the stolen states
                    for _ in range(self.shared_state.stolen_works.value-self.shared_state.assigned_works.value):
                        with self.shared_state.shared_queue_len.get_lock():
                            one_work = self.shared_state.shared_queue.get()
                            self.private_deque.append(one_work)
                            self.shared_state.shared_queue_len.value -= 1
                    assert self.shared_state.shared_queue_len.value == 0

                    with self.shared_state.workers_idle_status.get_lock():
                        for n, ele in enumerate(self.shared_state.workers_idle_status):
                            if ele == 0: # not idle
                                self.shared_state.workers_valid_status[n] = 1
                            else:
                                self.shared_state.workers_valid_status[n] = 0

                    self.shared_state.work_assign_ready.clear()
                    self.shared_state.reset_after_assgin()
                    self.shared_state.steal_assign_ready.set()


    def collect_results(self, s):
        """
        Conduct the function in terms of the configuration

        Parameters:
            s (FVIM or Flattice): an output reachable set

        """

        # Repair of the network model on the safety property
        if self.dnn.repair:
            unsafe_input_sets = self.dnn.backtrack(s)
            with self.shared_state.outputs_len.get_lock():
                if not self.shared_state.work_interrupted.is_set():
                    for aset in unsafe_input_sets:
                        unsafe_input = aset.vertices[[0]]
                        unsafe_output = np.dot(aset.vertices[0], s.M.T) + s.b.T
                        self.shared_state.outputs_len.value += 1
                        self.shared_state.outputs.put([unsafe_input, unsafe_output])
                        if self.shared_state.outputs_len.value >= self.output_len:
                            self.shared_state.work_interrupted.set()

        # Safety verification of DNN on the safety property
        elif self.dnn.verification:
            if not self.shared_state.work_interrupted.is_set():
                unsafe = self.dnn.verify(s)
                if unsafe:
                    self.shared_state.outputs.put(unsafe)
                    self.shared_state.work_interrupted.set()

        # Compute the exact unsafe input subspace on the safety property
        elif self.dnn.unsafe_inputd and (not self.dnn.exact_outputd):
            unsafe_inputs = self.dnn.backtrack(s)
            with self.shared_state.outputs_len.get_lock():
                if not self.shared_state.work_interrupted.is_set():
                    for aset in unsafe_inputs:
                        self.shared_state.outputs_len.value += 1
                        self.shared_state.outputs.put(aset)
                        if self.shared_state.outputs_len.value >= self.output_len:
                            self.shared_state.work_interrupted.set()

        # Compute the exact unsafe output reachable domain on the safety property
        elif (not self.dnn.unsafe_inputd) and self.dnn.exact_outputd:
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs_len.value += 1
                self.shared_state.outputs.put(s)

        # Compute the exact unsafe input subspace and exact unsafe output reachable domain
        elif self.dnn.unsafe_inputd and self.dnn.exact_outputd:
            unsafe_inputs = self.dnn.backtrack(s) # in FVIM or Flattice
            with self.shared_state.outputs_len.get_lock():
                self.shared_state.outputs.put([unsafe_inputs, s])
                self.shared_state.outputs_len.value += 1
        else:
            raise ValueError('Reachability configuration error!')


    def state_spawn_depth_first(self, tuple_state):
        """
        Depth-first computation of states

        Parameters:
            tuple_state (tuple): State
        """
        next_tuple_states = self.dnn.compute_state(tuple_state)
        if len(next_tuple_states) == 0:
            return
        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.collect_results(next_tuple_states[0][0])
            return

        if len(next_tuple_states) == 2:
            self.private_deque.append(next_tuple_states[1])

        self.state_spawn_depth_first(next_tuple_states[0])



    def state_spawn_breath_first(self, tuple_state):
        """
        Breath-first computation of states

        Parameters:
            tuple_state (tuple): State
        """
        next_tuple_states = self.dnn.compute_state(tuple_state)

        if len(next_tuple_states) == 0:
            return

        if next_tuple_states[0][1] == self.dnn._num_layer - 1: # last layer
            assert len(next_tuple_states) == 1
            self.collect_results(next_tuple_states[0][0])
            return

        for one_state in next_tuple_states:
            self.private_deque.append(one_state)


