from flask import current_app
from PIL import Image
import torch
from app.utils.helpers import load_document_indices
import logging
import tempfile
import base64
from io import BytesIO
import os
from byaldi import RAGMultiModalModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_minicpm_response(prompt, image_paths, device):
    try:
        tokenizer = current_app.config['TOKENIZER']
        model = current_app.config['MODEL']
        image_processor = current_app.config['IMAGE_PROCESSOR']
        RAG = current_app.config['RAG']

        logger.debug("Preparing the prompt for MiniCPM.")

        # Tokenize the prompt
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        # Process all images at once
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        processed_images = image_processor(images=images, return_tensors='pt')
        pixel_values = processed_images['pixel_values'].to(device)

        logger.debug(f"Processed images shape: {pixel_values.shape}")

        # Generate response from model
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up temporary files
        for image_path in image_paths:
            try:
                os.unlink(image_path)
                logger.info(f"Deleted temporary image file: {image_path}")
            except Exception as cleanup_e:
                logger.error(f"Failed to delete temporary image file {image_path}: {cleanup_e}", exc_info=True)

        return {
            "answer": answer,
            "tokens_consumed": len(input_ids[0])
        }

    except Exception as e:
        logger.error(f"Error in generate_minicpm_response: {e}", exc_info=True)
        raise

def query_document(doc_id, query, k=3):
    try:
        document_indices = load_document_indices()
        index_path = document_indices.get(doc_id)
        if not index_path:
            raise ValueError(f"No index found for document {doc_id}")

        logger.info(f"Index path for document {doc_id}: {index_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        
        logger.info(f"Index file size: {os.path.getsize(index_path)}")

        # Initialize the RAG model with the correct index
        logger.info("Initializing RAG model with the index")
        RAG = RAGMultiModalModel.from_index(index_path)
        
        # Perform the search
        logger.info(f"Performing search with query: {query}")
        rag_results = RAG.search(query, k=k)
        
        # Log the number of results
        logger.info(f"Number of results returned by byaldi: {len(rag_results)}")

        # Process results
        image_paths = []
        serializable_results = []
        for i, result in enumerate(rag_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Doc ID: {result.doc_id}")
            logger.info(f"  Page Number: {result.page_num}")
            logger.info(f"  Score: {result.score}")
            logger.info(f"  Metadata: {result.metadata}")
            
            # Check if there's an image associated with this result
            if hasattr(result, 'base64') and result.base64:
                logger.info(f"  Image found for result {i+1}")
                # Save image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
                    image_data = base64.b64decode(result.base64)
                    temp_image.write(image_data)
                    image_paths.append(temp_image.name)
                logger.info(f"  Image saved to: {temp_image.name}")
            else:
                logger.info(f"  No image found for result {i+1}")

            serializable_results.append({
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                "has_image": hasattr(result, 'base64') and bool(result.base64)
            })

        logger.info(f"Total number of images found: {len(image_paths)}")

        # Generate context and prompt
        context = "\n".join([f"Result {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])
        prompt = f"Based on the following search results, please answer this question: {query}\n\n{context}"

        # Generate response
        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        response = generate_minicpm_response(prompt, image_paths, device)

        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }

    except Exception as e:
        logger.error(f"Error in query_document: {str(e)}", exc_info=True)
        raise

def query_image(image, query, user_id):
    try:
        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image.save(temp_image.name)
            image_path = temp_image.name

        logger.info(f"Saved image to temporary path: {image_path}")

        # Initialize RAG model from app config
        RAG = current_app.config['RAG']
        
        # Perform the search
        rag_results = RAG.search(query, image_paths=[image_path], k=3)

        serializable_results = [
            {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                "base64": result.base64 if hasattr(result, 'base64') else None
            } for result in rag_results
        ]

        context = "\n".join([f"Image {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])
        prompt = f"Based on the following image descriptions, please answer this question: {query}\n\n{context}"

        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using device {device} for generating response.")
        response = generate_minicpm_response(prompt, [image_path], device)

        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }

    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        raise