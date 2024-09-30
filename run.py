import os
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from app.cuda_init import init_cuda, initialize_llm, initialize_rag, cleanup
from app import create_app
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def initialize_model(rank):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        ).to(f'cuda:{rank}')
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

def run_app(rank, world_size):
    try:
        setup(rank, world_size)
        init_cuda()
        model, tokenizer, image_processor = initialize_model(rank)
        RAG = initialize_rag(rank, world_size)

        app = create_app()
        app.config['RANK'] = rank
        app.config['WORLD_SIZE'] = world_size
        app.model = model
        app.tokenizer = tokenizer
        app.image_processor = image_processor
        app.config['RAG'] = RAG

        app.run(host='0.0.0.0', port=5000 + rank)
    except Exception as e:
        logger.error(f"Error in process {rank}: {e}")
    finally:
        cleanup()
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    mp.set_start_method('spawn', force=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        logger.error("No GPUs available. Exiting.")
        sys.exit(1)
    mp.spawn(run_app, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()