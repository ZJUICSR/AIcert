"""
These functions are used to collect and share states between workers

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import multiprocessing as mp
import numpy as np
import copy as cp

class SharedState:
    """
    A Class for the communication between workers

        Attributes:
            num_workers (int): Number of local workers
            shared_queue (queue): Shared queue to store stolen states
            shared_queue_len (int): Length of the shared_queue
            outputs (queue): Results storage
            outputs_len (int): Length of the outputs
            initial_steal_assign (mp.Event): Indicator whether the operation is done
            initial_completed_workers (mp.Value): Number of workers accomplished
            stolen_works (mp.Value): Number of stolen states from local non-idle workers
            assigned_works (mp.Value): Number of assigned states to idle workers
            num_empty_assign (mp.Value): Number of idle workers that are assigned with no states
            works_to_assign_per_worker (mp.Value): Number of states that should be assigned to each worker
            num_workers_done (mp.Value): Number of workers whose computation is done
            work_steal_ready (mp.Event): Indicator whether the work stealing from non-idle workers is ready
            work_assign_ready (mp.Event): Indicator whether the work assigning to idle workers is ready
            steal_assign_ready (mp.Event): Indicator whether the steal&assign mode is ready
            work_done (mp.Event): Indicator whether the computation is done
            work_interrupted (mp.Event): Indicator of work interruption
            all_workers_done (mp.Event): Indicator the computation in all the workers is done
            work_steal_rate (mp.Value): Rate of the local states to steal
            workers_valid_status (mp.Array): Indicators of valid workers to steal states from
            workers_idle_status (mp.Array): Indicators of idle workers
            workers_to_assign (mp.Array): Indicators of workers to assign with states
            workers_assigned (mp.Array): Indicators of workers that are assigned with states

        Methods:
            initialize_shared_queue(prop):
                Initialize the shared queue with input sets from the safety property
            increase_queue_len(num):
                Increase the length of the shared queue
            decrease_queue_len():
                Decrease the length of the shared queue
            compute_steal_rate():
                Compute the rate of states to steal from local workers
            reset_after_assgin():
                Reset indicators after stealing & assigning states

    """
    def __init__(self, safety_property, num_workers):

        self.num_workers = num_workers
        self.shared_queue = mp.Manager().Queue()
        self.shared_queue_len = mp.Value('i', 0)

        self.outputs = mp.Manager().Queue()
        self.outputs_len = mp.Value('i', 0)

        self.initial_steal_assign = mp.Event()
        self.initial_completed_workers = mp.Value('i', 0)
        self.initialize_shared_queue(safety_property)

        self.stolen_works = mp.Value('i', 0)
        self.assigned_works = mp.Value('i', 0)
        self.num_empty_assign = mp.Value('i', 0)
        self.works_to_assign_per_worker = mp.Value('i', 0)
        self.num_workers_done = mp.Value('i', 0)

        self.work_steal_ready = mp.Event()
        self.work_assign_ready = mp.Event()
        self.steal_assign_ready = mp.Event()
        self.work_done = mp.Event()
        self.work_interrupted = mp.Event()
        self.all_workers_done = mp.Event()

        self.work_steal_rate = mp.Value('f', 0.0)

        self.workers_valid_status = mp.Array('i',[1]*num_workers)
        self.workers_idle_status = mp.Array('i', [0]*num_workers)
        self.workers_to_assign = mp.Array('i', [0]*num_workers)
        self.workers_assigned = mp.Array('i', [0]*num_workers)


    def initialize_shared_queue(self, prop):
        """
        Initialize the shared queue with input sets from the safety property

        Parameters:
            prop (Property): Safety property

        """
        self.shared_queue.put((cp.deepcopy(prop.input_set), -1, np.array([])))
        self.increase_queue_len(1)


    def increase_queue_len(self, num):
        """
        Increase the length of the shared queue

        Parameters:
            num (int): Number to increase

        """
        with self.shared_queue_len.get_lock():
            self.shared_queue_len.value += num


    def decrease_queue_len(self):
        """
        Decrease the length of the shared queue

        """
        with self.shared_queue_len.get_lock():
            self.shared_queue_len.value -= 1


    def compute_steal_rate(self):
        """
        Compute the rate of states to steal from local workers
        """
        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = sum(self.workers_to_assign)/self.num_workers


    def reset_after_assgin(self):
        """
        Reset indicators after stealing & assigning states
        """
        if not self.work_done.is_set():
            assert not self.work_steal_ready.is_set()
            assert not self.work_assign_ready.is_set()
            assert self.shared_queue_len.value == 0

        if not (self.work_done.is_set() or self.work_interrupted.is_set()):
            assert self.shared_queue.empty()

        with self.work_steal_rate.get_lock():
            self.work_steal_rate.value = 0.0

        with self.stolen_works.get_lock():
            self.stolen_works.value = 0

        with self.assigned_works.get_lock():
            self.assigned_works.value = 0

        with self.workers_assigned.get_lock():
            for n,_ in enumerate(self.workers_assigned):
                self.workers_assigned[n] = 0

        with self.workers_to_assign.get_lock():
            for n,_ in enumerate(self.workers_to_assign):
                self.workers_to_assign[n] = 0









