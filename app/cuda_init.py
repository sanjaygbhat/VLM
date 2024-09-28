import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def init_cuda():
    if torch.cuda.is_available():
        print("Initializing CUDA settings...")
        torch.cuda.empty_cache()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use two GPUs
        print("CUDA settings initialized.")
    else:
        print("CUDA is not available. Using CPU.")

def get_gpu_memory_usage():
    return [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]

def initialize_llm(rank, world_size):
    setup(rank, world_size)
    
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    return model, tokenizer

def run_model(rank, world_size):
    model, tokenizer = initialize_llm(rank, world_size)
    # Your model usage code here
    cleanup()

def init_distributed_model():
    world_size = torch.cuda.device_count()
    mp.spawn(run_model, args=(world_size,), nprocs=world_size, join=True)
