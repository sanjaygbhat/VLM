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

                # Log the types of each key in processed
                for key, value in processed.items():
                    logger.debug(f"Processed key: {key}, Type: {type(value)}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")

                pixel_values = processed.get('pixel_values', None)

                if pixel_values is None:
                    logger.error("Processed 'pixel_values' is missing.")
                    raise ValueError("Image processing failed: 'pixel_values' not found.")

                logger.debug(f"Raw pixel_values type: {type(pixel_values)}")
                logger.debug(f"Raw pixel_values shape: {pixel_values.shape if isinstance(pixel_values, torch.Tensor) else 'Unknown'}")

                # Ensure pixel_values is a tensor
                if isinstance(pixel_values, list):
                    try:
                        pixel_values = torch.stack(pixel_values).to(device)
                        logger.debug("Converted 'pixel_values' from list to tensor.")
                    except Exception as stack_e:
                        logger.error(f"Failed to stack 'pixel_values' list into tensor: {str(stack_e)}")
                        logger.error(f"pixel_values list content: {[type(v) for v in pixel_values]}")
                        raise

                logger.debug(f"Final pixel_values shape: {pixel_values.shape}")

            except Exception as img_process_e:
                logger.error(f"Failed to process images: {str(img_process_e)}")
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
        logger.debug(f"Deleted temporary image file: {image_path}")

        return {
            "results": serializable_results,
            "answer": response["answer"],
            "tokens_consumed": response["tokens_consumed"]
        }
    except Exception as e:
        logger.error(f"Error in query_image: {str(e)}", exc_info=True)
        raise