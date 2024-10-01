from flask import current_app
from PIL import Image
import torch
from app.utils.helpers import load_document_indices
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_minicpm_response(prompt, image_path, device):
    try:
        tokenizer = current_app.config['TOKENIZER']
        model = current_app.config['MODEL']
        image_processor = current_app.config['IMAGE_PROCESSOR']

        logger.debug("Preparing the prompt for MiniCPM.")
        messages = [{"role": "user", "content": prompt}]
        minicpm_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Initialize pixel_values as None
        pixel_values = None

        if image_path:
            logger.debug(f"Processing image at path: {image_path}")
            image = Image.open(image_path).convert("RGB")
            # Process the image here (resize, normalize, etc.)
            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
            logger.debug("Image processing complete.")
        else:
            # If no image, create a dummy tensor of the correct shape
            logger.debug("No image provided. Creating dummy pixel_values.")
            pixel_values = torch.zeros((1, 3, 224, 224), device=device)

        # Proceed with generating the response using the model
        logger.debug("Generating response with MiniCPM.")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=minicpm_prompt['input_ids'].to(device),
                attention_mask=minicpm_prompt['attention_mask'].to(device),
                pixel_values=pixel_values,
                max_length=current_app.config.get('MAX_LENGTH', 20),
                do_sample=current_app.config.get('DO_SAMPLE', False),
                num_beams=current_app.config.get('NUM_BEAMS', 1),
                early_stopping=current_app.config.get('EARLY_STOPPING', False)
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug("Response generation complete.")

        return {
            "response": response
        }
    except Exception as e:
        logger.error(f"Error in generate_minicpm_response: {str(e)}", exc_info=True)
        raise

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
        logger.debug(f"Using device {device} for generating response.")
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
    try:
        image_path = image.filename
        image.save(image_path)
        logger.info(f"Saved image to path: {image_path}")

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
        logger.debug(f"Using device {device} for generating response.")
        response = generate_minicpm_response(prompt, image_path, device)

        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }
    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        raise