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

        logger.debug("Preparing the prompt for MiniCPM.")

        # Tokenize the prompt
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        # Process images in batch
        images = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))  # Ensure image is in RGB
            except Exception as img_open_e:
                logger.error(f"Failed to open image {image_path}: {str(img_open_e)}")

        if images:
            logger.debug(f"Number of images loaded: {len(images)}")
            try:
                # Process all images together
                processed = image_processor(images, return_tensors="pt")
                logger.debug(f"Processed image_processor output keys: {processed.keys()}")

                pixel_values = processed.get('pixel_values', None)

                if pixel_values is None:
                    logger.error("Processed 'pixel_values' is missing.")
                    raise ValueError("Image processing failed: 'pixel_values' not found.")

                # Check if pixel_values is a list
                if isinstance(pixel_values, list):
                    # Ensure each element is a tensor
                    pixel_values_tensors = []
                    for idx, pv in enumerate(pixel_values):
                        if isinstance(pv, torch.Tensor):
                            pixel_values_tensors.append(pv)
                        else:
                            logger.error(f"pixel_values[{idx}] is not a Tensor.")
                            raise TypeError(f"pixel_values[{idx}] is not a Tensor.")
                    # Stack tensors
                    pixel_values = torch.stack(pixel_values_tensors).to(device)
                    logger.debug("Converted pixel_values from list to tensor.")
                elif isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(device)
                    logger.debug(f"pixel_values is already a tensor. Type: {type(pixel_values)} and shape: {pixel_values.shape}")
                else:
                    logger.error(f"Unexpected type for pixel_values: {type(pixel_values)}")
                    raise TypeError(f"Unexpected type for pixel_values: {type(pixel_values)}")

                # Ensure pixel_values is not empty
                if pixel_values.size(0) == 0:
                    logger.error("pixel_values tensor is empty.")
                    raise ValueError("Image processing failed: 'pixel_values' tensor is empty.")

            except Exception as stack_e:
                logger.error(f"Failed to stack pixel_values list into tensor: {str(stack_e)}")
                raise

        else:
            logger.warning("No images provided.")
            pixel_values = None

        # Prevent model generation if pixel_values is invalid
        if pixel_values is None:
            logger.error("pixel_values is None. Cannot generate response without valid image data.")
            raise ValueError("Image processing failed. 'pixel_values' is None.")

        # Generate response from the model
        logger.debug("Generating response with MiniCPM.")
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )

        # Decode the response
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug("Response generated successfully.")

        return {
            "answer": response_text,
            "tokens_consumed": output.size(1)
        }

    except Exception as e:
        logger.error(f"Error in generate_minicpm_response: {e}", exc_info=True)
        raise

def query_document(doc_id, query, k=3):
    try:
        # Load document indices
        document_indices = load_document_indices()
        index_path = document_indices.get(doc_id, None)
        if not index_path:
            logger.error(f"Index not found for document ID: {doc_id}")
            raise ValueError("Document index not found.")

        # Initialize RAG model from app config
        RAG = current_app.config['RAG']
        
        # Perform the search
        rag_results = RAG.search(query, index_path=index_path, k=k)

        # Prepare the results
        serializable_results = [
            {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                "base64": result.base64
            } for result in rag_results
        ]

        context = "\n".join([f"Document {i+1} (Page {result['page_num']}):\n{result['metadata']}" for i, result in enumerate(serializable_results)])
        prompt = f"Based on the following documents, please answer this question: {query}\n\n{context}"

        # Determine device
        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using device {device} for generating response.")
        response = generate_minicpm_response(prompt, [], device)  # No images in this context

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
                "base64": result.base64
            } for result in rag_results
        ]

        context = "\n".join([f"Image {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])
        prompt = f"Based on the following image descriptions, please answer this question: {query}\n\n{context}"

        # Determine device
        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using device {device} for generating response.")
        response = generate_minicpm_response(prompt, [image_path], device)

        # Clean up the temporary file
        os.unlink(image_path)

        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }
    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        raise