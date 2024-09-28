from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.document_service import upload_document

bp = Blueprint('document', __name__)

@bp.route('/upload_pdf', methods=['POST'])
@jwt_required()
def upload_pdf():
    current_user = get_jwt_identity()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        upload_result = upload_document(file, current_user)
        if upload_result is None:
            return jsonify({"error": "Failed to process document"}), 500
        doc_id, indexing_time = upload_result
        return jsonify({"document_id": doc_id, "indexing_time": indexing_time}), 200
    return jsonify({"error": "Invalid file type"}), 400