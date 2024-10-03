from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MiniCPM:
    def __init__(self, model_name="openbmb/MiniCPM-V-2_6", device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map="auto"
        )
        self.device = device
        self.model.eval()
        logger.info(f"MiniCPM model '{model_name}' loaded on {self.device}.")

    def generate_response(self, input_text, max_length=150):
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("MiniCPM generated a response successfully.")
            return response
        except Exception as e:
            logger.error(f"Error in MiniCPM.generate_response: {str(e)}", exc_info=True)
            return "An error occurred while generating the response."