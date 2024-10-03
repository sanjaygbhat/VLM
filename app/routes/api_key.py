from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.api_key_service import APIKeyService
from app.services.payment_service import PaymentService
from app.models.user import User
from app import db

bp = Blueprint('api_key', __name__)
payment_service = PaymentService()

@bp.route('/generate_api_key', methods=['POST'])
@jwt_required()
def generate_api_key():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    api_key = APIKeyService.create_api_key(user_id)
    return jsonify({"api_key": api_key.key, "credits": user.credits}), 201

@bp.route('/replenish_credits', methods=['POST'])
@jwt_required()
def replenish_credits():
    data = request.get_json()
    amount = data.get('amount', 0)
    payment_details = data.get('payment_details')  # e.g., credit card info

    if not amount or not payment_details:
        return jsonify({"error": "Missing parameters."}), 400

    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    # Process payment
    if payment_service.process_payment(user, amount, payment_details):
        # Credits are already added in the payment service
        return jsonify({"message": "Credits replenished successfully.", "credits": user.credits}), 200
    else:
        return jsonify({"error": "Payment processing failed."}), 402  # Payment Required