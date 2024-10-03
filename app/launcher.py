import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
from byaldi import RAGMultiModalModel
from app.models.minicpm_model import MiniCPM

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
    
    # Initialize MiniCPM with the specific CUDA device
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    minicpm = MiniCPM(device=device)
    
    # Initialize the RAG model
    try:
        rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
        logger.info("RAGMultiModalModel initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAGMultiModalModel: {e}")
        sys.exit(1)

    # Create and configure Flask app
    from app import create_app

    app = create_app()
    app.config['RAG'] = rag_model
    app.config['MINICPM'] = minicpm
    app.config['TOKENIZER'] = minicpm.tokenizer

    # Run the Flask app
    port = 5000 + rank
    app.run(host='0.0.0.0', port=port)
    logger.info(f"Process {rank}: Flask app running on port {port}.")

def start_launcher():
    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        logger.error("No GPUs available. Exiting.")
        sys.exit(1)
    logger.info(f"Starting application with {world_size} GPU(s).")
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)