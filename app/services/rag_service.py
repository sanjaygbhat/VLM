from flask import current_app
from PIL import Image
import torch
from app.utils.helpers import load_document_indices
import logging
import tempfile
from base64 import b64decode
from io import BytesIO
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_minicpm_response(prompt, image_paths, device):
    """
    Generates a response using the MiniCPM model based on the given prompt and images.

    Args:
        prompt (str): The text prompt to generate a response for.
        image_paths (list of str): List of image file paths to include in the prompt.
        device (torch.device): The device to run the model on.

    Returns:
        dict: A dictionary containing the generated answers and tokens consumed.
    """
    try:
        tokenizer = current_app.config['TOKENIZER']
        model = current_app.config['MODEL']
        image_processor = current_app.config['IMAGE_PROCESSOR']

        logger.debug("Preparing the prompt for MiniCPM.")

        if image_paths:
            k = len(image_paths)
            logger.debug(f"Number of images to process: {k}")

            # Tokenize the prompt
            tokens = tokenizer(prompt, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            # Repeat input_ids and attention_mask k times to align with images
            input_ids = input_ids.repeat(k, 1)
            attention_mask = attention_mask.repeat(k, 1)

            # Process each image and collect pixel_values
            pixel_values_list = []
            for idx, image_path in enumerate(image_paths):
                logger.debug(f"Processing image {idx+1}/{k} at path: {image_path}")
                try:
                    image = Image.open(image_path).convert("RGB")
                    pixel_values = image_processor(images=image, return_tensors='pt')['pixel_values'].to(device)
                    pixel_values_list.append(pixel_values)
                except Exception as img_e:
                    logger.error(f"Failed to process image {image_path}: {img_e}")
                    # Optionally, append a dummy tensor or handle accordingly
                    pixel_values_list.append(torch.zeros((1, 3, 224, 224), device=device))

            # Concatenate pixel_values into a single tensor
            pixel_values = torch.cat(pixel_values_list, dim=0)  # Shape: (k, C, H, W)

            logger.debug("All images processed successfully.")

        else:
            # If no images are provided, create dummy pixel_values to satisfy model requirements
            logger.debug("No images provided. Creating dummy pixel_values.")
            pixel_values = torch.zeros((1, 3, 224, 224), device=device)
            tokens = tokenizer(prompt, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            k = 1  # Batch size

        logger.debug(f"Input IDs shape: {input_ids.shape}")
        logger.debug(f"Pixel values shape: {pixel_values.shape}")

        # Generate the response using the model
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the generated tokens to strings
        if k > 1:
            answers = [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(outputs.size(0))]
            logger.debug(f"Generated {len(answers)} answers.")
        else:
            answers = [tokenizer.decode(outputs[0], skip_special_tokens=True)]
            logger.debug("Generated 1 answer.")

        # Calculate tokens consumed
        tokens_consumed = input_ids.size(1)

        return {
            "answers": answers,
            "tokens_consumed": tokens_consumed
        }

    except Exception as e:
        logger.error(f"Error in generate_minicpm_response: {e}", exc_info=True)
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

        # Extract images from the top k results
        image_paths = []
        temp_files = []

        for i, res in enumerate(serializable_results):
            if res.get('base64'):
                try:
                    image_data = b64decode(res['base64'])
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                except Exception as img_e:
                    logger.error(f"Failed to decode image for result {i}: {img_e}")
                    continue

                # Save to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                try:
                    image.save(temp_file.name)
                    image_paths.append(temp_file.name)
                    temp_files.append(temp_file.name)  # Keep track for cleanup
                    logger.debug(f"Saved image for result {i} to {temp_file.name}")
                except Exception as save_e:
                    logger.error(f"Failed to save image for result {i}: {save_e}")
                    temp_file.close()
                    os.unlink(temp_file.name)
            else:
                logger.warning(f"Result {i} has no 'base64' image data.")

        if not image_paths:
            logger.error("No images found in the top results.")
            raise ValueError("No images available for the given query.")

        # Build context from metadata
        context = "\n".join([f"Excerpt {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])

        prompt = f"Based on the following excerpts and images, please answer this question: {query}\n\n{context}"

        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using device {device} for generating response.")

        # Generate the response with image_paths
        response = generate_minicpm_response(prompt, image_paths, device)

        # Clean up temporary image files
        for path in temp_files:
            try:
                os.unlink(path)
                logger.debug(f"Deleted temporary image file: {path}")
            except Exception as del_e:
                logger.warning(f"Failed to delete temporary image file {path}: {del_e}")

        # Aggregate answers if multiple
        if len(response['answers']) == 1:
            answer = response['answers'][0]
        else:
            # Example aggregation: concatenate all answers
            answer = "\n".join(response['answers'])

        return {
            "results": serializable_results,
            "answer": answer,
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