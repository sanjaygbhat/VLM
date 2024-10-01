from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from config import Config
import logging

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')  # Ensure you have a Config class

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Import models
    from app.models.user import User
    from app.models.document import Document

    # Import and register blueprints
    from app.routes import auth, document, query
    app.register_blueprint(auth.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(query.bp)

    return app