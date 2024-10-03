
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
import logging
from app.models.minicpm_model import MiniCPM

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

def initialize_model(rank, world_size):
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process {rank}: Using device {device}")

    try:
        # Initialize MiniCPM
        minicpm = MiniCPM(model_name=MODEL_NAME, device=device)
        logger.debug(f"Process {rank}: MiniCPM model initialized on {device}.")

        # Initialize the Image Processor
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        logger.debug(f"Process {rank}: Image processor initialized.")

        return minicpm, image_processor

    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize MiniCPM or Image Processor - {e}", exc_info=True)
        raise

def cleanup():
    torch.cuda.empty_cache()
    logger.debug("Cleared CUDA cache.")