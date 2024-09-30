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

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Initialize tokenizer, model, and image processor with trust_remote_code=True
    try:
        app.tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        app.model = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        ).to('cuda')
        app.image_processor = AutoImageProcessor.from_pretrained(
            "openbmb/MiniCPM-V-2_6",
            trust_remote_code=True
        )
        logger.info("Model, tokenizer, and image processor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model components: {e}")
        raise

    # Import models
    from app.models.user import User
    from app.models.document import Document

    # Import and register blueprints
    from app.routes import auth, document, query
    app.register_blueprint(auth.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(query.bp)

    return app