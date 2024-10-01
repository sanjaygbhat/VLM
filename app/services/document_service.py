import time
import logging
import os
import uuid
import hashlib
from werkzeug.utils import secure_filename
from app.services.rag_service import query_document
from app import Config, db
from app.utils.helpers import load_document_indices, save_document_indices
from app.models.document import Document
from flask import current_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    """Calculate SHA256 hash of file contents from a file path"""
    start_time = time.time()
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    hash_time = time.time() - start_time
    logger.info(f"File hash calculation took {hash_time:.2f} seconds")
    return hasher.hexdigest()

def upload_document(file, user_id):
    start_time = time.time()
    
    # Step 1: Save the file
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(Config.UPLOAD_FOLDER, f"{doc_id}_{filename}")
    file.save(file_path)
    logger.info(f"Step 1: File saving took {time.time() - start_time:.2f} seconds")

    # Step 2: Calculate file hash
    start_time = time.time()
    file_hash = get_file_hash(file_path)
    logger.info(f"Step 2: File hashing took {time.time() - start_time:.2f} seconds")

    # Optional Step: Check for duplicate file_hash
    existing_document = Document.query.filter_by(file_hash=file_hash).first()
    if existing_document:
        logger.info(f"Duplicate file detected: {existing_document.id}")
        return existing_document.id, 0  # Assuming no indexing time for duplicates

    # Step 3: Create document record with file_hash
    start_time = time.time()
    document = Document(id=doc_id, filename=filename, file_hash=file_hash, user_id=user_id)
    db.session.add(document)
    db.session.commit()
    logger.info(f"Step 3: Document record creation took {time.time() - start_time:.2f} seconds")

    # Step 4: Create index name
    start_time = time.time()
    index_name = f"index_{doc_id}"
    index_path = os.path.join(Config.INDEX_FOLDER, f"{index_name}.faiss")
    logger.info(f"Step 4: Index preparation took {time.time() - start_time:.2f} seconds")

    # Step 5: Indexing using RAG
    start_time = time.time()
    try:
        # Initialize RAG model from app config
        RAG = current_app.config['RAG']
        # Index the document
        RAG.index(
            input_path=file_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        db.session.delete(document)
        db.session.commit()
        return None
    indexing_time = time.time() - start_time
    logger.info(f"Step 5: Indexing took {indexing_time:.2f} seconds")

    # Step 6: Move index
    start_time = time.time()
    byaldi_index_path = os.path.join('.byaldi', index_name)
    if os.path.exists(byaldi_index_path):
        os.rename(byaldi_index_path, index_path)
        logger.info(f"Moved index from {byaldi_index_path} to {index_path}")
    else:
        logger.error(f"Index not found at {byaldi_index_path}")
        db.session.delete(document)
        db.session.commit()
        return None
    logger.info(f"Step 6: Moving index took {time.time() - start_time:.2f} seconds")

    # Step 7: Update document indices
    start_time = time.time()
    document_indices = load_document_indices()
    document_indices[doc_id] = index_path
    save_document_indices(document_indices)
    logger.info(f"Step 7: Updating document indices took {time.time() - start_time:.2f} seconds")

    # Step 8: Save to database (already done in Step 3)

    return doc_id, indexing_time