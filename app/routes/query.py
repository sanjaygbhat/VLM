from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.rag_service import query_document
from app.models.document import Document
from PIL import Image
import base64
from io import BytesIO
import torch

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
        # Get results from byaldi
        byaldi_results = query_document(doc_id, query, k)
        
        # Process byaldi results for MiniCPM
        processed_results = process_byaldi_results(byaldi_results)
        
        # Get MiniCPM response
        minicpm_response = query_minicpm(query, processed_results)
        
        return jsonify({
            "byaldi_results": byaldi_results,
            "minicpm_response": minicpm_response
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the query."}), 500

def process_byaldi_results(results):
    processed_results = []
    for result in results:
        if result.get('base64'):
            # Convert base64 to PIL Image
            img_data = base64.b64decode(result['base64'])
            img = Image.open(BytesIO(img_data)).convert('RGB')
            processed_results.append(img)
    return processed_results

def query_minicpm(query, images):
    model = current_app.config['MODEL']
    tokenizer = current_app.config['TOKENIZER']
    device = current_app.config['DEVICE']

    # Ensure the model is in evaluation mode and on the correct device
    model = model.eval().to(device)

    # Prepare the message for the model
    msgs = [{'role': 'user', 'content': images + [query]}]

    # Generate the answer
    with torch.no_grad():
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )

    return answer

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