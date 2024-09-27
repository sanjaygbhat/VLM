import os
import uuid
from werkzeug.utils import secure_filename
from app.services.rag_service import RAG
from app import Config
from app.utils.helpers import load_document_indices, save_document_indices

def upload_document(file, user_id):
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(Config.UPLOAD_FOLDER, f"{doc_id}_{filename}")
    file.save(file_path)
    
    index_name = f"index_{doc_id}"
    index_path = os.path.join(Config.INDEX_FOLDER, f"{index_name}.faiss")
    
    RAG.index(
        input_path=file_path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True
    )
    
    byaldi_index_path = os.path.join('.byaldi', index_name)
    if os.path.exists(byaldi_index_path):
        os.rename(byaldi_index_path, index_path)
        print(f"Moved index from {byaldi_index_path} to {index_path}")
    else:
        print(f"Index not found at {byaldi_index_path}")
        return None
    
    document_indices = load_document_indices()
    document_indices[doc_id] = index_path
    save_document_indices(document_indices)
    
    return doc_id