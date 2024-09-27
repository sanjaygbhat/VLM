import os
import uuid
import hashlib
from werkzeug.utils import secure_filename
from app.services.rag_service import RAG
from app import Config, db
from app.utils.helpers import load_document_indices, save_document_indices
from app.models.document import Document

def get_file_hash(file):
    """Calculate SHA256 hash of file contents"""
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file.read(4096), b""):
        hasher.update(chunk)
    file.seek(0)  # Reset file pointer to beginning
    return hasher.hexdigest()

def upload_document(file, user_id):
    file_hash = get_file_hash(file)
    
    # Check if document with this hash already exists
    existing_doc = Document.query.filter_by(file_hash=file_hash).first()
    if existing_doc:
        return existing_doc.id
    
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
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

    new_doc = Document(id=doc_id, filename=filename, file_hash=file_hash, user_id=user_id)
    db.session.add(new_doc)
    db.session.commit()
    
    return doc_id