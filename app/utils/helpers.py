import json
import os
from app import Config

def load_document_indices():
    if os.path.exists(Config.DOCUMENT_INDEX_FILE):
        with open(Config.DOCUMENT_INDEX_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_document_indices(indices):
    with open(Config.DOCUMENT_INDEX_FILE, 'w') as f:
        json.dump(indices, f)