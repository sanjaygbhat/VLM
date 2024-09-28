import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from byaldi import RAGMultiModalModel

def init_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available. Using CPU.")

def get_gpu_memory_usage():
    return [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]

def initialize_llm(rank, world_size):
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"
    
    # Assign a specific GPU to this process
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Process {rank}: Using device {device}")
    
    # Load the model directly to the assigned device
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    return model, tokenizer

def initialize_rag(rank, world_size):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Process {rank}: Loading RAG model to device {device}")
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    RAG = RAG.to(device)  # Ensure RAG supports .to(device)
    return RAG

def cleanup():
    torch.cuda.empty_cache()