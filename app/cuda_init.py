import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_cuda():
    if torch.cuda.is_available():
        print("Initializing CUDA settings...")
        torch.cuda.empty_cache()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Use two GPUs
        print("CUDA settings initialized.")
    else:
        print("CUDA is not available. Using CPU.")

def get_gpu_memory_usage():
    return [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]

def initialize_llm(rank, world_size):
    MODEL_NAME = "openbmb/MiniCPM-V-2_6"
    
    # Create a device map to distribute the model across GPUs
    device_map = {
        'llm.model.embed_tokens': 0,
        'llm.model.norm': world_size - 1,
        'llm.lm_head': world_size - 1,
        'vpm': 0,  # Assign vision model to first GPU
        'resampler': world_size - 1  # Assign resampler to last GPU
    }
    
    # Distribute LLM layers
    llm_layers = 28  # Number of LLM layers from the model printout
    layers_per_gpu = llm_layers // world_size
    for i in range(llm_layers):
        device_map[f'llm.model.layers.{i}'] = i // layers_per_gpu
    
    # Load the model with the custom device map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=device_map,
        max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.8 / 1024**3)}GiB" for i in range(world_size)}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    return model, tokenizer

def cleanup():
    torch.cuda.empty_cache()

# Remove the run_model and init_distributed_model functions
