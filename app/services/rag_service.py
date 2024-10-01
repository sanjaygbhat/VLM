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

        logger.debug(f"Preparing the prompt for MiniCPM. Prompt: {prompt[:100]}...")
        logger.debug(f"Number of image paths: {len(image_paths)}")

        # Tokenize the prompt
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        logger.debug(f"Tokenized prompt shape: {input_ids.shape}")

        # Process images in batch
        images = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    img_rgb = img.convert('RGB')
                    images.append(img_rgb)
                logger.debug(f"Loaded image from {image_path}, size: {img_rgb.size}, mode: {img_rgb.mode}")
            except Exception as img_open_e:
                logger.error(f"Failed to open image {image_path}: {str(img_open_e)}")

        if images:
            logger.debug(f"Number of images loaded: {len(images)}")
            try:
                # Process all images together
                processed = image_processor(images, return_tensors="pt")
                logger.debug(f"Processed image_processor output keys: {processed.keys()}")
                logger.debug(f"Processed image_processor output types: {[type(v) for v in processed.values()]}")

                pixel_values = processed.get('pixel_values', None)

                if pixel_values is None:
                    logger.error("Processed 'pixel_values' is missing.")
                    raise ValueError("Image processing failed: 'pixel_values' not found.")

                logger.debug(f"Raw pixel_values type: {type(pixel_values)}")
                logger.debug(f"Raw pixel_values shape: {pixel_values.shape if isinstance(pixel_values, torch.Tensor) else [v.shape for v in pixel_values] if isinstance(pixel_values, list) else 'Unknown'}")

                # Ensure pixel_values is a tensor
                if isinstance(pixel_values, list):
                    try:
                        pixel_values = torch.stack(pixel_values).to(device)
                        logger.debug("Converted 'pixel_values' from list to tensor.")
                    except Exception as stack_e:
                        logger.error(f"Failed to stack 'pixel_values' list into tensor: {str(stack_e)}")
                        logger.error(f"pixel_values list content: {[type(v) for v in pixel_values]}")
                        raise
                elif isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(device)
                    logger.debug(f"pixel_values is already a tensor. Shape: {pixel_values.shape}")
                else:
                    logger.error(f"Unexpected type for pixel_values: {type(pixel_values)}")
                    raise TypeError(f"Unexpected type for pixel_values: {type(pixel_values)}")

                logger.debug(f"Final pixel_values shape: {pixel_values.shape}")
                logger.debug(f"Final pixel_values device: {pixel_values.device}")
            except Exception as img_proc_e:
                logger.error(f"Failed to process images: {str(img_proc_e)}")
                pixel_values = None
        else:
            pixel_values = None
            logger.debug("No images to process.")

        if pixel_values is None:
            logger.error("pixel_values is None. Cannot generate response without valid image data.")
            raise ValueError("Image processing failed. 'pixel_values' is None.")

        # Generate response
        try:
            logger.debug("Generating response with model...")
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )
            logger.debug(f"Model output shape: {output.shape}")
        except Exception as gen_e:
            logger.error(f"Error during model.generate: {str(gen_e)}")
            raise

        # Decode the output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Decoded response length: {len(response)}")

        return {
            "answer": response,
            "tokens_consumed": len(output[0])
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
        
        # Use the RAG model from app config
        RAG = current_app.config['RAG']
        RAG_specific = RAG.from_index(index_path)

        logger.info(f"Performing search with query: {query}")
        results = RAG_specific.search(query, k=k)

        # Process the results
        serializable_results = []
        image_paths = []
        for i, result in enumerate(results):
            serializable_result = {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                # "base64": result.base64  # Optional: Include if needed
            }
            
            # Handle image data if present
            if result.base64:
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(result.base64)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # Save image to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        image.save(temp_file, format='JPEG')
                        image_paths.append(temp_file.name)
                    
                    serializable_result["image_path"] = temp_file.name
                except Exception as img_e:
                    logger.error(f"Failed to process image for result {i}: {str(img_e)}")
            
            serializable_results.append(serializable_result)

        # Generate context from metadata
        context = "\n".join([f"Excerpt {i+1}:\n{result['metadata']}" for i, result in enumerate(serializable_results)])

        prompt = f"Based on the following excerpts and images, please answer this question: {query}\n\n{context}"

        device = torch.device(f'cuda:{current_app.config["RANK"]}' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using device {device} for generating response.")

        # Generate the response with image_paths
        response = generate_minicpm_response(prompt, image_paths, device)

        # Clean up temporary image files
        for path in image_paths:
            try:
                os.unlink(path)
                logger.debug(f"Deleted temporary image file: {path}")
            except Exception as del_e:
                logger.warning(f"Failed to delete temporary image file {path}: {del_e}")

        return {
            "results": serializable_results,
            "answer": response["answer"],  # Corrected key
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