from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.rag_service import query_document, query_minicpm, query_image
from app.models.document import Document
from app.utils.security import api_key_required
from PIL import Image
import base64
from io import BytesIO
import torch

bp = Blueprint('query', __name__)

@bp.route('/query', methods=['POST'])
@jwt_required(optional=True)
@api_key_required
def query_doc():
    # Determine the user
    if hasattr(request, 'user'):
        user = request.user
    else:
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({"error": "Authentication required."}), 401
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found."}), 404

    data = request.json
    doc_id = data.get('document_id')
    query = data.get('query')
    k = data.get('k', 3)

    if not doc_id or not query:
        return jsonify({"error": "Missing document_id or query"}), 400

    # Verify that the document belongs to the current user
    document = Document.query.filter_by(id=doc_id, user_id=user.id).first()
    if not document:
        return jsonify({"error": "Document not found or access denied."}), 404

    try:
        # Get results from byaldi
        byaldi_results = query_document(doc_id, query, k)

        # Process byaldi results for MiniCPM
        processed_results = query_minicpm(query, byaldi_results)

        return jsonify({"answer": processed_results}), 200

    except Exception as e:
        current_app.logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your query."}), 500

@bp.route('/query_image', methods=['POST'])
@jwt_required(optional=True)
@api_key_required
def query_img():
    # Determine the user
    if hasattr(request, 'user'):
        user = request.user
    else:
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({"error": "Authentication required."}), 401
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found."}), 404

    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    image = request.files['image']
    query = request.form.get('query')

    if not query:
        return jsonify({"error": "Missing query"}), 400

    if image and image.filename != '':
        try:
            # Process image and query with MiniCPM
            results = query_image(image, query, user.id)
            return jsonify(results), 200
        except Exception as e:
            current_app.logger.error(f"Error processing image query: {e}", exc_info=True)
            return jsonify({"error": "An error occurred while processing the image query."}), 500
    return jsonify({"error": "Invalid image file"}), 400