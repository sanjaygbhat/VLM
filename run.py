import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import logging

# Import initialization functions for CUDA and RAG
from app.cuda_init import initialize_rag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"GPU {i} memory usage: {memory:.2f} GB")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def initialize_model(rank, world_size):
    try:
        # Use mixed precision
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Implement more granular model parallelism
        total_layers = 32  # Adjust this based on the actual number of layers in MiniCPM-V-2_6
        layers_per_gpu = total_layers // world_size
        start_layer = rank * layers_per_gpu
        end_layer = start_layer + layers_per_gpu if rank != world_size - 1 else total_layers

        device_map = {f"transformer.layers.{i}": rank for i in range(start_layer, end_layer)}
        device_map["transformer.embed_tokens"] = 0  # Corrected mapping
        device_map["transformer.norm"] = world_size - 1
        device_map["lm_head"] = world_size - 1

        model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
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
        return model, tokenizer, image_processor
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize model - {e}")
        raise

def cleanup():
    dist.destroy_process_group()

def run_app(rank, world_size):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        
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

        # Initialize RAG and add to app.config
        RAG = initialize_rag(rank, world_size)
        app.config['RAG'] = RAG

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