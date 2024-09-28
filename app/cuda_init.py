import multiprocessing
import os
import torch.multiprocessing as mp

import torch
from vllm import LLM, SamplingParams


def init_cuda():
    if mp.get_start_method(allow_none=True) != 'spawn':
        print("Initializing CUDA settings...")
        mp.set_start_method('spawn', force=True)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use two GPUs
        print("CUDA settings initialized.")

def get_gpu_memory_usage():
    return [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]
        
def initialize_llm():
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        tensor_parallel_size=2,
        dtype="float16",
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        enforce_eager=True,
    )
    return llm
