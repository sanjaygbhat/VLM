import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import logging

# Import initialization functions for CUDA and RAG
from app.cuda_init import initialize_rag

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

def initialize_model(rank, world_size):
    try:
        # Use mixed precision
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.debug(f"Process {rank}: Using dtype {dtype}.")

        # Load the model and its configuration
        model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            torch_dtype=dtype,
            device_map=None,  # We'll handle device mapping manually
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        config = model.config
        logger.debug(f"Process {rank}: Loaded model and configuration.")

        # Log all available attributes in the configuration
        config_attrs = dir(config)
        logger.debug(f"Process {rank}: Available config attributes: {config_attrs}")

        # Correctly access the transformer layers
        if hasattr(config, 'llm') and hasattr(config.llm, 'model'):
            total_layers = len(config.llm.model.layers)
            logger.info(f"Process {rank}: Total transformer layers found: {total_layers}")
        else:
            logger.error(f"Process {rank}: 'MiniCPMVConfig' does not have 'llm.model.layers'")
            raise AttributeError("'MiniCPMVConfig' does not have 'llm.model.layers'")

        # Calculate layers per GPU
        layers_per_gpu = total_layers // world_size
        start_layer = rank * layers_per_gpu
        end_layer = start_layer + layers_per_gpu if rank != world_size - 1 else total_layers
        logger.debug(f"Process {rank}: Assigning layers {start_layer} to {end_layer} to GPU {rank}.")

        # Create device_map based on the correct attribute
        device_map = {
            f"llm.model.layers.{i}": rank for i in range(start_layer, end_layer)
        }
        device_map["llm.model.embed_tokens"] = 0  # Assign embedding layer to GPU 0
        device_map["llm.model.norm"] = world_size - 1
        device_map["lm_head"] = world_size - 1

        logger.debug(f"Process {rank}: Device map: {device_map}")

        # Load model with the updated device_map
        model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Process {rank}: Model loaded with device mapping.")

        tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        logger.info(f"Process {rank}: Tokenizer and image processor initialized.")

        # Log all module names to ensure correct mapping
        for name, module in model.named_modules():
            logger.debug(f"Process {rank}: Module Name: {name}")

        return model, tokenizer, image_processor

    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize model - {e}", exc_info=True)
        raise

def cleanup():
    dist.destroy_process_group()
    logger.debug("Destroyed distributed process group.")

def run_app(rank, world_size):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        logger.debug(f"Process {rank}: Set CUDA device to {rank}.")

        logger.info(f"Process {rank}: Before model initialization")
        log_gpu_memory()

        model, tokenizer, image_processor = initialize_model(rank, world_size)

        logger.info(f"Process {rank}: After model initialization")
        log_gpu_memory()

        # Initialize CUDA
        torch.cuda.empty_cache()
        logger.debug(f"Process {rank}: Cleared CUDA cache.")

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
        logger.debug(f"Process {rank}: RAG model initialized and added to app config.")

        app.run(host='0.0.0.0', port=5000 + rank)
        logger.info(f"Process {rank}: Flask app running on port {5000 + rank}.")

    except Exception as e:
        logger.error(f"Error in process {rank}: {e}", exc_info=True)
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