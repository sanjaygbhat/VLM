from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.rag_service import query_document, query_image
from app.models.document import Document

bp = Blueprint('query', __name__)

@bp.route('/query', methods=['POST'])
@jwt_required()
def query_doc():
    user_id = get_jwt_identity()
    data = request.json
    doc_id = data.get('document_id')
    query = data.get('query')
    k = data.get('k', 3)
    
    if not doc_id or not query:
        return jsonify({"error": "Missing document_id or query"}), 400
    
    # Verify that the document belongs to the current user
    document = Document.query.filter_by(id=doc_id, user_id=user_id).first()
    if not document:
        return jsonify({"error": "Document not found or access denied."}), 404
    
    try:
        results = query_document(doc_id, query, k)
        return jsonify(results), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the query."}), 500

@bp.route('/query_image', methods=['POST'])
@jwt_required()
def query_img():
    user_id = get_jwt_identity()
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    image = request.files['image']
    query = request.form.get('query')
    
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    if image and image.filename != '':
        try:
            # Assuming image contains information to link to a document
            # You might need to modify this based on your actual implementation
            results = query_image(image, query, user_id)
            return jsonify(results), 200
        except Exception as e:
            return jsonify({"error": "An error occurred while processing the image query."}), 500
    return jsonify({"error": "Invalid image file"}), 400