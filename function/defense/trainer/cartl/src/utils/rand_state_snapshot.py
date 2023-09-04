from typing import Optional, Union, List, Dict
import torch
import numpy as np
import copy


"""
    Here are stuffs for supporting deterministic experiments. Including getting 
    and setting random states from/for 'numpy', 'torch', 'torch.cuda', etc.
"""
## GETTING RNG STATE ##
def snapshotting_numpy_rand_state():
    return copy.deepcopy(np.random.get_state())

def snapshotting_torch_rand_state()->np.ndarray:
    bt = copy.deepcopy(torch.random.get_rng_state()) #type:torch.ByteTensor
    return bt.numpy()

def snapshotting_cuda_rand_state(device:Optional[Union[torch.device, str]]="cuda")->np.ndarray:
    if not torch.cuda.is_available():
        return np.array([])
    if isinstance(device, str):
        device = torch.device(device)
    bt = copy.deepcopy(torch.cuda.get_rng_state(device)) #type:torch.ByteTensor
    return bt.numpy()

def snapshotting_all_cuda_rand_state()->List[np.ndarray]:
    if not torch.cuda.is_available():
        return []
    bts = torch.cuda.get_rng_state_all() #type:List[torch.ByteTensor]
    return [copy.deepcopy(bt).numpy() for bt in bts]


## SETTING RNG STATE ##
def set_numpy_rand_state(state):
    np.random.set_state(state)

def set_torch_rand_state(state:np.ndarray):
    torch.random.set_rng_state(torch.from_numpy(state))

def set_cuda_rand_state(state:np.ndarray, device:Optional[Union[torch.device, str]]="cuda"):
    if torch.cuda.is_available():
        if isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_rng_state(torch.from_numpy(state), torch.device(device))

# Use with caution. Manually setting states for all GPUs may mess up others' experiments.
def set_all_cuda_rand_state(states:List[np.ndarray]):
    if not torch.cuda.is_available():
        return
    if len(states) != torch.cuda.device_count():
        raise ValueError("The number of given states should be equal to the number of GPUs")
    torch.cuda.set_rng_state_all([torch.from_numpy(st) for st in states])



class RandStateSnapshooter(object):
    @staticmethod
    def lazy_take(path):
        import pickle as pkl
        with open(path, "wb") as f:
            pkl.dump(RandStateSnapshooter.take_snapshot(), f)
    
    @staticmethod
    def lazy_set(path):
        import pickle as pkl
        with open(path, "rb") as f:
            states = pkl.load(f)
        RandStateSnapshooter.set_rand_state(states)
        
    @staticmethod
    def take_snapshot():
        state = {
            "numpy": snapshotting_numpy_rand_state(),
            "torch": snapshotting_torch_rand_state(),
            "cuda": snapshotting_cuda_rand_state()
        }
        return state

    @staticmethod
    def set_rand_state(state:Dict):
        if "numpy" in state:
            set_numpy_rand_state(state["numpy"])
        
        if "torch" in state:
            set_torch_rand_state(state["torch"])
        
        if "cuda" in state:
            set_cuda_rand_state(state["cuda"])
    