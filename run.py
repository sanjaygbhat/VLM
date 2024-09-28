import os
import sys
import torch.multiprocessing as mp
from torch.distributed import run

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.cuda_init import init_cuda
from app import create_app

def run_app(rank, world_size):
    init_cuda()
    app = create_app()
    app.config['RANK'] = rank
    app.config['WORLD_SIZE'] = world_size
    app.run(host='0.0.0.0', port=5000 + rank)  # Use different ports for each process

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()