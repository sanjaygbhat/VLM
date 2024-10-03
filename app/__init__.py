from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor
from app.config import Config
import logging

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')  # Ensure you have a Config class

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Import models
    from app.models.user import User
    from app.models.document import Document
    from app.models.api_key import APIKey  # Import the new APIKey model

    # Import and register blueprints
    from app.routes import auth, document, query, api_key
    app.register_blueprint(auth.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(query.bp)
    app.register_blueprint(api_key.bp)  # Register the APIKey blueprint

    return app