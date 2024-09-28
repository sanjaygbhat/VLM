import time
import logging
import os
import uuid
import hashlib
from werkzeug.utils import secure_filename
from app.services.rag_service import query_document  # Update import as RAG is now handled in rag_service
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
    start_time = time.time()
    
    # Step 1: Save the file
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(Config.UPLOAD_FOLDER, f"{doc_id}_{filename}")
    file.save(file_path)
    logger.info(f"Step 1: File saving took {time.time() - start_time:.2f} seconds")

    # Step 2: Create document record
    start_time = time.time()
    document = Document(id=doc_id, filename=filename, user_id=user_id)
    db.session.add(document)
    db.session.commit()
    logger.info(f"Step 2: Document record creation took {time.time() - start_time:.2f} seconds")

    # Step 3: Create index name
    start_time = time.time()
    index_name = f"index_{doc_id}"
    index_path = os.path.join(Config.INDEX_FOLDER, f"{index_name}.faiss")
    logger.info(f"Step 3: Index preparation took {time.time() - start_time:.2f} seconds")

    # Step 4: Indexing using RAG
    start_time = time.time()
    try:
        indexing_results = query_document(doc_id, "Indexing", k=1)  # Replace "Indexing" with appropriate dummy query if needed
        # Assuming RAG.index is replaced by query_document for indexing
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return None
    indexing_time = time.time() - start_time
    logger.info(f"Step 4: Indexing took {indexing_time:.2f} seconds")

    # Step 5: Move index
    start_time = time.time()
    byaldi_index_path = os.path.join('.byaldi', index_name)
    if os.path.exists(byaldi_index_path):
        os.rename(byaldi_index_path, index_path)
        logger.info(f"Moved index from {byaldi_index_path} to {index_path}")
    else:
        logger.error(f"Index not found at {byaldi_index_path}")
        return None
    logger.info(f"Step 5: Moving index took {time.time() - start_time:.2f} seconds")

    # Step 6: Update document indices
    start_time = time.time()
    document_indices = load_document_indices()
    document_indices[doc_id] = index_path
    save_document_indices(document_indices)
    logger.info(f"Step 6: Updating document indices took {time.time() - start_time:.2f} seconds")

    # Step 7: Save to database (already done in Step 2)

    return doc_id, indexing_time