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
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logging
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

        # Initialize the full device_map dynamically
        # This approach ensures that all modules are correctly mapped
        logger.debug("Initializing device map dynamically.")

        # First, load the configuration to get module names
        config = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).config

        # Assume 'layers' are named appropriately in the config
        total_layers = len(config.transformer.layers)
        logger.info(f"Total transformer layers found in config: {total_layers}")
        layers_per_gpu = total_layers // world_size
        logger.debug(f"Layers per GPU: {layers_per_gpu}")

        device_map = {}

        for gpu in range(world_size):
            start_layer = gpu * layers_per_gpu
            # Ensure the last GPU takes any remaining layers
            end_layer = start_layer + layers_per_gpu if gpu != world_size - 1 else total_layers
            for layer_idx in range(start_layer, end_layer):
                layer_name = f"llm.model.layers.{layer_idx}"
                device_map[layer_name] = gpu
                logger.debug(f"Assigning {layer_name} to GPU {gpu}.")

        # Assign embedding, norm, and head layers
        device_map["llm.model.embed_tokens"] = 0  # Embedding layer to GPU 0
        device_map["llm.model.norm"] = world_size - 1  # Norm layer to last GPU
        device_map["lm_head"] = world_size - 1  # Output head to last GPU
        logger.debug(f"Assigned embed_tokens, norm, and lm_head to GPUs.")

        logger.debug(f"Final device map: {device_map}")

        # Now, load the model with the constructed device_map
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
        logger.debug(f"Process {rank}: Tokenizer initialized.")

        image_processor = AutoImageProcessor.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        logger.debug(f"Process {rank}: Image processor initialized.")

        logger.info(f"Process {rank}: Model, tokenizer, and image processor initialized successfully.")

        # Optional: Log module names for verification
        logger.debug("Listing all module names in the model:")
        for name, module in model.named_modules():
            logger.debug(f"Module Name: {name}")

        layer_names = [name for name, module in model.named_modules() if 'layers.' in name]
        logger.info(f"Process {rank}: Total transformer layers found: {len(layer_names)}")

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