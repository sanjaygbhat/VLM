import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    
    # Calculate the number of layers to put on each GPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.transformer.h)
    layers_per_gpu = num_layers // world_size
    
    # Assign layers to each GPU
    start_layer = rank * layers_per_gpu
    end_layer = start_layer + layers_per_gpu if rank < world_size - 1 else num_layers
    
    # Create a device map to distribute the model across GPUs
    device_map = {f'transformer.h.{i}': rank for i in range(start_layer, end_layer)}
    device_map['transformer.wte'] = 0  # Embedding layer on first GPU
    device_map['transformer.ln_f'] = world_size - 1  # Final layer norm on last GPU
    device_map['lm_head'] = world_size - 1  # Language model head on last GPU
    
    # Load the model with the custom device map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=device_map,
        max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.8 / 1024**3)}GiB" for i in range(world_size)}
    )
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    return model, tokenizer

def cleanup():
    dist.destroy_process_group()

# Remove the run_model and init_distributed_model functions
