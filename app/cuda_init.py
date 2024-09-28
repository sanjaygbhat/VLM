import multiprocessing
import os
import torch.multiprocessing as mp

def init_cuda():
    print("Initializing CUDA settings...")
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use two GPUs
    print("CUDA settings initialized.")