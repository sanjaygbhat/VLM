import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.cuda_init import init_cuda, initialize_llm, initialize_rag, cleanup
from app import create_app

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_app(rank, world_size):
    try:
        setup(rank, world_size)
        init_cuda()
        model, tokenizer = initialize_llm(rank, world_size)
        RAG = initialize_rag(rank, world_size)
        
        app = create_app()
        app.config['RANK'] = rank
        app.config['WORLD_SIZE'] = world_size
        app.model = model
        app.tokenizer = tokenizer
        app.config['RAG'] = RAG
        app.run(host='0.0.0.0', port=5000 + rank)
    except Exception as e:
        print(f"Error in process {rank}: {e}")
    finally:
        cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPUs available. Exiting.")
        sys.exit(1)
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()