import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///your_database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    INDEX_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indices')
    DOCUMENT_INDEX_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'document_indices.json')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')