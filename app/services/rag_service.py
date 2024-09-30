from flask import current_app
from PIL import Image
import torch
from app.utils.helpers import load_document_indices
import logging

logger = logging.getLogger(__name__)

def generate_minicpm_response(prompt, image_path, device):
    tokenizer = current_app.tokenizer
    model = current_app.model
    
    messages = [{"role": "user", "content": prompt}]
    minicpm_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(minicpm_prompt, return_tensors="pt").to(device)
    
    # Initialize pixel_values as None
    pixel_values = None
    
    if image_path:
        image = Image.open(image_path).convert("RGB")
        # Process the image here (you might need to resize, normalize, etc.)
        # This is a placeholder - replace with actual image processing code
        pixel_values = current_app.image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    else:
        # If no image, create a dummy tensor of the correct shape
        # You might need to adjust the shape based on your model's requirements
        pixel_values = torch.zeros((1, 3, 224, 224), device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=pixel_values,
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

import logging

logger = logging.getLogger(__name__)

def query_document(doc_id, query, k=3):
    try:
        document_indices = load_document_indices()
        if doc_id not in document_indices:
            raise ValueError(f"Invalid document_id: {doc_id}")
        
        index_path = document_indices[doc_id]
        logger.info(f"Loading index from path: {index_path}")
        RAG_specific = current_app.config['RAG'].from_index(index_path)
        
        logger.info(f"Performing search with query: {query}")
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
        
        context = "\n".join([f"Excerpt {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])
        prompt = f"Based on the following excerpts, please answer this question: {query}\n\n{context}"
        
        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Generating response using device: {device}")
        response = generate_minicpm_response(prompt, None, device)  # Pass None for image_path
        
        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }
    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}", exc_info=True)
        raise

def query_image(image, query):
    image_path = image.filename
    image.save(image_path)
    
    RAG_specific = current_app.config['RAG'].from_index(index_path=image_path)
    rag_results = RAG_specific.search(query, image_path=image_path)
    
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in rag_results
    ]
    
    context = "\n".join([f"Image {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])
    prompt = f"Based on the following image descriptions, please answer this question: {query}\n\n{context}"
    
    device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
    response = generate_minicpm_response(prompt, image_path, device)
    
    return {
        "results": serializable_results,
        "answer": response["answer"],
        "tokens_consumed": response["tokens_consumed"]
    }