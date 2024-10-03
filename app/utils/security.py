from functools import wraps
from flask import request, jsonify, current_app
from app.services.api_key_service import APIKeyService

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key_str = request.headers.get('x-api-key')
        if not api_key_str:
            return jsonify({"error": "API key is missing."}), 401

        api_key = APIKeyService.validate_api_key(api_key_str)
        if not api_key:
            return jsonify({"error": "Invalid or inactive API key."}), 401

        # Deduct credits
        if not APIKeyService.deduct_credits(api_key, amount=1):
            return jsonify({"error": "Insufficient credits."}), 402  # Payment Required

        # Attach user to request context
        request.user = api_key.user
        return f(*args, **kwargs)
    return decorated