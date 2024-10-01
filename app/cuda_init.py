import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from byaldi import RAGMultiModalModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_cuda():
    if torch.cuda.is_available():
        logger.info("CUDA is available.")
    else:
        logger.warning("CUDA is not available. Using CPU.")

def get_gpu_memory_usage():
    usage = [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]
    logger.debug(f"GPU memory usage: {usage}")
    return usage

def initialize_llm(rank, world_size):
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"

    # Assign a specific GPU to this process
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process {rank}: Using device {device}")

    # Load the model directly to the assigned device
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map={"llm.model.layers": device.index}  # Placeholder, managed in run.py
        )
        model.to(device)  # Ensure model is moved to the device
        logger.debug(f"Process {rank}: LLM model moved to {device}.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        logger.debug(f"Process {rank}: Tokenizer initialized.")

        return model, tokenizer
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize LLM - {e}", exc_info=True)
        raise

def initialize_rag(rank, world_size):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process {rank}: Loading RAG model to device {device}")

    try:
        # Initialize RAG model without using .to(device)
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
        logger.debug(f"Process {rank}: RAG model loaded.")

        # If RAGMultiModalModel handles device assignment internally, ensure it uses the correct device
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            logger.debug(f"Process {rank}: Set CUDA device for RAG to {device}.")

        return RAG
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize RAG model - {e}", exc_info=True)
        raise

def cleanup():
    torch.cuda.empty_cache()
    logger.debug("Cleared CUDA cache.")