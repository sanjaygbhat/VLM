import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import logging
import re
from byaldi import RAGMultiModalModel  # Add this import

# Import initialization functions for CUDA
from app.cuda_init import initialize_model  # Make sure this import exists

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

def generate_device_map(model, world_size, rank):
    """
    Dynamically generates a device map based on the model's named modules.
    Assigns transformer layers evenly across available GPUs and maps other modules appropriately.
    """
    device_map = {}
    layer_pattern = re.compile(r'^llm\.model\.layers\.(\d+)$')
    layers = []

    # Collect all transformer layer module names
    for name, module in model.named_modules():
        if layer_pattern.match(name):
            layers.append(name)

    total_layers = len(layers)
    logger.info(f"Process {rank}: Total transformer layers found: {total_layers}")

    # Calculate layers per GPU
    layers_per_gpu = total_layers // world_size
    extra_layers = total_layers % world_size

    current_layer = 0
    for device_rank in range(world_size):
        num_layers = layers_per_gpu + (1 if device_rank < extra_layers else 0)
        assigned_layers = layers[current_layer:current_layer + num_layers]
        for layer_name in assigned_layers:
            device_map[layer_name] = f'cuda:{device_rank}'
        current_layer += num_layers

    logger.info(f"Process {rank}: Device map generated.")
    return device_map

def initialize_model(rank, world_size):
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process {rank}: Using device {device}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map=generate_device_map(AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True), world_size, rank)
        )
        model.to(device)
        logger.debug(f"Process {rank}: LLM model moved to {device}.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        logger.debug(f"Process {rank}: Tokenizer initialized.")

        # Initialize the Image Processor
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        logger.debug(f"Process {rank}: Image processor initialized.")

        return model, tokenizer, image_processor

    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize model - {e}", exc_info=True)
        raise

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

    # Initialize RAG and add to app.config
    try:
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
        app.config['RAG'] = RAG
        logger.debug(f"Process {rank}: RAG model initialized and added to app config.")
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize RAG model - {e}", exc_info=True)
        raise

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