from app import db
from datetime import datetime, timedelta
import secrets

class APIKey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(days=365))
    is_active = db.Column(db.Boolean, default=True)

    def generate_new_key(self):
        self.key = secrets.token_hex(32)
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(days=365)
        self.is_active = True
