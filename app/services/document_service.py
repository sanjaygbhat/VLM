import time
import logging
import os
import uuid
import hashlib
from werkzeug.utils import secure_filename
from app.services.rag_service import RAG
from app import Config, db
from app.utils.helpers import load_document_indices, save_document_indices
from app.models.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_hash(file):
    """Calculate SHA256 hash of file contents"""
    start_time = time.time()
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file.read(4096), b""):
        hasher.update(chunk)
    file.seek(0)  # Reset file pointer to beginning
    hash_time = time.time() - start_time
    logger.info(f"File hash calculation took {hash_time:.2f} seconds")
    return hasher.hexdigest()

def upload_document(file, user_id):
    overall_start_time = time.time()
    
    # Step 1: Calculate file hash
    start_time = time.time()
    file_hash = get_file_hash(file)
    logger.info(f"Step 1: File hash calculation took {time.time() - start_time:.2f} seconds")

    # Step 2: Check for existing document
    start_time = time.time()
    existing_doc = Document.query.filter_by(file_hash=file_hash).first()
    if existing_doc:
        logger.info(f"Step 2: Existing document check took {time.time() - start_time:.2f} seconds")
        logger.info(f"Document already exists. Returning existing ID: {existing_doc.id}")
        return existing_doc.id
    logger.info(f"Step 2: Existing document check took {time.time() - start_time:.2f} seconds")

    # Step 3: Prepare file path
    start_time = time.time()
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(Config.UPLOAD_FOLDER, f"{doc_id}_{filename}")
    logger.info(f"Step 3: File path preparation took {time.time() - start_time:.2f} seconds")

    # Step 4: Save file
    start_time = time.time()
    file.save(file_path)
    logger.info(f"Step 4: File saving took {time.time() - start_time:.2f} seconds")

    # Step 5: Prepare indexing
    start_time = time.time()
    index_name = f"index_{doc_id}"
    index_path = os.path.join(Config.INDEX_FOLDER, f"{index_name}.faiss")
    logger.info(f"Step 5: Index preparation took {time.time() - start_time:.2f} seconds")

    # Step 6: Indexing
    start_time = time.time()
    RAG.index(
        input_path=file_path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=True
    )
    indexing_time = time.time() - start_time
    logger.info(f"Step 6: Indexing took {indexing_time:.2f} seconds")

    # Step 7: Move index
    start_time = time.time()
    byaldi_index_path = os.path.join('.byaldi', index_name)
    if os.path.exists(byaldi_index_path):
        os.rename(byaldi_index_path, index_path)
        logger.info(f"Moved index from {byaldi_index_path} to {index_path}")
    else:
        logger.error(f"Index not found at {byaldi_index_path}")
        return None
    logger.info(f"Step 7: Moving index took {time.time() - start_time:.2f} seconds")

    # Step 8: Update document indices
    start_time = time.time()
    document_indices = load_document_indices()
    document_indices[doc_id] = index_path
    save_document_indices(document_indices)
    logger.info(f"Step 8: Updating document indices took {time.time() - start_time:.2f} seconds")

    # Step 9: Save to database
    start_time = time.time()
    new_doc = Document(id=doc_id, filename=filename, file_hash=file_hash, user_id=user_id)
    db.session.add(new_doc)
    db.session.commit()
    logger.info(f"Step 9: Saving to database took {time.time() - start_time:.2f} seconds")

    total_time = time.time() - overall_start_time
    logger.info(f"Total upload_document process took {total_time:.2f} seconds")
    
    return doc_id