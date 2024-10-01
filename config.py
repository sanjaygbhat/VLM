import os
import torch
from app.utils.image_processor import ImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///your_database.db'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    
    # File Storage Configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    INDEX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indices')
    DOCUMENT_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'document_indices.json')
    DEVICE = device
    IMAGE_PROCESSOR = ImageProcessor(device)
    

