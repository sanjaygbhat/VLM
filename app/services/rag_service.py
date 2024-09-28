import os
import subprocess
import base64
import torch
from byaldi import RAGMultiModalModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
from app import Config
from app.utils.helpers import load_document_indices
from werkzeug.utils import secure_filename

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

MODEL_NAME = "openbmb/MiniCPM-V-2_6"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Clear CUDA cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / (1024 * 1024)

print(f"GPU memory usage before LLM init: {get_gpu_memory_usage():.2f} MB")

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    tensor_parallel_size=2,
    dtype="float16",
    max_num_batched_tokens=4096,
    max_num_seqs=64,
    enforce_eager=True,
)

def get_gpu_memory_usage():
    return [torch.cuda.memory_allocated(i) / (1024 * 1024) for i in range(torch.cuda.device_count())]

print(f"GPU memory usage after LLM init: {get_gpu_memory_usage():.2f} MB")

def generate_minicpm_response(prompt, image_path):
    messages = [{"role": "user", "content": prompt}]
    minicpm_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = {
        "prompt": minicpm_prompt,
        "multi_modal_data": {
            "image": Image.open(image_path).convert("RGB")
        },
    }

    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    sampling_params = SamplingParams(
        stop_token_ids=stop_token_ids, 
        temperature=0.7,
        top_p=0.95,
        max_tokens=256
    )

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    return {
        "answer": outputs[0].outputs[0].text,
        "tokens_consumed": {
            "prompt_tokens": len(tokenizer.encode(minicpm_prompt)),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
            "total_tokens": len(tokenizer.encode(minicpm_prompt)) + len(outputs[0].outputs[0].token_ids)
        }
    }

def query_document(doc_id, query, k):
    document_indices = load_document_indices()
    if doc_id not in document_indices:
        return {"error": "Invalid document_id"}
    
    index_path = document_indices[doc_id]
    
    if not os.path.exists(index_path):
        return {"error": "Index file not found"}
    
    RAG_specific = RAGMultiModalModel.from_index(index_path)
    results = RAG_specific.search(query, k=k)
    
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in results
    ]
    
    prompt = "Here are some relevant document excerpts:\n\n"
    prompt += "\n".join([f"Excerpt {idx}:\nMetadata: {result['metadata']}\n" 
                         for idx, result in enumerate(serializable_results, 1)])
    prompt += f"\nBased on these excerpts, please answer the following question: {query}"

    image_path = os.path.join(Config.UPLOAD_FOLDER, f"{doc_id}.png")
    minicpm_response = generate_minicpm_response(prompt, image_path)
    
    return {
        "results": serializable_results,
        **minicpm_response
    }

def query_image(image, query):
    filename = secure_filename(image.filename)
    image_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    image.save(image_path)
    
    rag_results = RAG.search(query, image_path=image_path)
    
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in rag_results
    ]
    
    with open(image_path, "rb") as image_file:
        encoded_query_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = "Here's the query image and some relevant image results:\n\n"
    prompt += "\n".join([f"Image {idx}:\nMetadata: {result['metadata']}\n" 
                         for idx, result in enumerate(serializable_results, 1)])
    prompt += f"\nBased on these images, please answer the following question: {query}"

    minicpm_response = generate_minicpm_response(prompt, image_path)
    
    os.remove(image_path)
    
    return {
        "results": serializable_results,
        "query_image_base64": encoded_query_image,
        **minicpm_response
    }

import subprocess

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / (1024 * 1024)

# Add this line just before LLM initialization
print(f"GPU memory usage before LLM init: {get_gpu_memory_usage():.2f} MB")