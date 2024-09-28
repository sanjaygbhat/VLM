from ..cuda_init import init_cuda, get_gpu_memory_usage
init_cuda()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from app import Config
from app.utils.helpers import load_document_indices
from byaldi import RAG  # Import RAG from byaldi

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"GPU memory usage before model init: {get_gpu_memory_usage()}")
# Remove the init_distributed_model() call
print(f"GPU memory usage after model init: {get_gpu_memory_usage()}")

def generate_minicpm_response(prompt, image_path):
    messages = [{"role": "user", "content": prompt}]
    minicpm_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(minicpm_prompt, return_tensors="pt").to('cuda')
    image = Image.open(image_path).convert("RGB")
    
    # Assuming the model has been initialized with a custom device map
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "answer": response,
        "tokens_consumed": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(outputs[0])
        }
    }

def query_document(doc_id, query, k=3):
    # Implement document querying logic here
    pass

def query_image(image, query):
    # Implement image querying logic here
    pass