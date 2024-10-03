from app.models.api_key import APIKey
from app.models.user import User
from app import db
import secrets
from datetime import datetime

class APIKeyService:
    @staticmethod
    def create_api_key(user_id):
        key = secrets.token_hex(32)
        api_key = APIKey(
            key=key,
            user_id=user_id
        )
        db.session.add(api_key)
        db.session.commit()
        return api_key

    @staticmethod
    def validate_api_key(key):
        api_key = APIKey.query.filter_by(key=key, is_active=True).first()
        if api_key and api_key.expires_at > datetime.utcnow():
            return api_key
        return None

    @staticmethod
    def deduct_credits(api_key, amount=1):
        user = api_key.user
        if user.credits >= amount:
            user.credits -= amount
            db.session.commit()
            return True
        return False

    @staticmethod
    def replenish_credits(user, amount):
        user.credits += amount
        db.session.commit()