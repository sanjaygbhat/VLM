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
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Inspect the model structure
    print(model)
    
    # Create a simple device map to distribute the model across GPUs
    device_map = {f'model.layers.{i}': i % world_size for i in range(len(model.model.layers))}
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = world_size - 1
    device_map['lm_head'] = world_size - 1
    
    # Load the model with the custom device map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map=device_map,
        max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.8 / 1024**3)}GiB" for i in range(world_size)}
    )
    
    # Move the model to the current device
    model.to(rank)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    return model, tokenizer

def cleanup():
    dist.destroy_process_group()

# Remove the run_model and init_distributed_model functions
