import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import logging
from byaldi import RAGMultiModalModel

# Import initialization functions for CUDA
from app.cuda_init import initialize_model

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"GPU {i} memory usage: {memory:.2f} GB")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logger.debug(f"Process {rank}: Initialized distributed process group.")

def run_app(rank, world_size):
    setup(rank, world_size)
    model, tokenizer, image_processor = initialize_model(rank, world_size)

    # Create and configure Flask app
    from app import create_app

    app = create_app()
    app.config['RANK'] = rank
    app.config['WORLD_SIZE'] = world_size
    app.config['MODEL'] = model
    app.config['TOKENIZER'] = tokenizer
    app.config['IMAGE_PROCESSOR'] = image_processor

    # Run the Flask app
    port = 5000 + rank
    app.run(host='0.0.0.0', port=port)
    logger.info(f"Process {rank}: Flask app running on port {port}.")

def main():
    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        logger.error("No GPUs available. Exiting.")
        sys.exit(1)
    logger.info(f"Starting application with {world_size} GPU(s).")
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()