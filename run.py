import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from flask import Flask
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.memory_allocated(i) / 1e6  # in MB
        logger.info(f"GPU {i} memory usage: {memory:.2f} MB")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def initialize_model(rank, world_size):
    try:
        # Use mixed precision
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load the model with device_map for multi-GPU
        model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",  # Automatically map layers to available GPUs
            low_cpu_mem_usage=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )

        image_processor = AutoImageProcessor.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )

        logger.info(f"Process {rank}: Model, tokenizer, and image processor initialized successfully.")
        log_gpu_memory()
        return model, tokenizer, image_processor
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize model - {e}")
        raise

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")

def run_app(rank, world_size):
    try:
        setup(rank, world_size)
        logger.info(f"Process {rank}: Before model initialization")
        log_gpu_memory()

        model, tokenizer, image_processor = initialize_model(rank, world_size)

        logger.info(f"Process {rank}: After model initialization")
        log_gpu_memory()

        # Initialize CUDA
        torch.cuda.empty_cache()

        # Create and configure Flask app
        from app import create_app  # Ensure create_app is defined in app/__init__.py

        app = create_app()
        app.config['RANK'] = rank
        app.config['WORLD_SIZE'] = world_size
        app.config['MODEL'] = model
        app.config['TOKENIZER'] = tokenizer
        app.config['IMAGE_PROCESSOR'] = image_processor

        app.run(host='0.0.0.0', port=5000 + rank)
    except Exception as e:
        logger.error(f"Error in process {rank}: {e}")
    finally:
        cleanup()

def main():
    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        logger.error("No GPUs available. Exiting.")
        sys.exit(1)
    logger.info(f"Starting application with {world_size} GPUs.")
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()