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
    logger.info(f"Document saved to {file_path}")

    # Step 2: Calculate file hash
    file_hash = get_file_hash(file_path)

    # Step 3: Check for duplicate
    existing_doc = Document.query.filter_by(file_hash=file_hash).first()
    if existing_doc:
        logger.warning("Duplicate document found. Deleting the uploaded file.")
        os.remove(file_path)
        return None  # Or return existing_doc.id, etc.

    # Step 4: Create document record
    document = Document(
        id=doc_id,
        filename=filename,
        file_hash=file_hash,
        user_id=user_id
    )
    db.session.add(document)
    db.session.commit()
    logger.info(f"Document record created with ID {doc_id}")

    # Step 5: Index the document with byaldi
    index_name = f"index_{doc_id}"
    index_path = os.path.join(Config.INDEX_FOLDER, index_name)
    try:
        # Assuming byaldi has a function to create index
        query_document.create_index(file_path, index_path, index_name)
        logger.info(f"Document indexed at {index_path}")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        db.session.delete(document)
        db.session.commit()
        return None

    indexing_time = time.time() - start_time
    logger.info(f"Document upload and indexing took {indexing_time:.2f} seconds")

    # Step 6: Update document indices
    start_time = time.time()
    document_indices = load_document_indices()
    document_indices[doc_id] = index_path
    save_document_indices(document_indices)
    logger.info(f"Updated document indices in {time.time() - start_time:.2f} seconds")

    return doc_id, indexing_time