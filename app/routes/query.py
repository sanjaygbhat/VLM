from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.rag_service import query_document, query_image

bp = Blueprint('query', __name__)

@bp.route('/query', methods=['POST'])
@jwt_required()
def query_doc():
    data = request.json
    doc_id = data.get('document_id')
    query = data.get('query')
    k = data.get('k', 3)
    
    if not doc_id or not query:
        return jsonify({"error": "Missing document_id or query"}), 400
    
    results = query_document(doc_id, query, k)
    return jsonify(results), 200

@bp.route('/query_image', methods=['POST'])
@jwt_required()
def query_img():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    image = request.files['image']
    query = request.form.get('query')
    
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    results = query_image(image, query)
    return jsonify(results), 200