from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import Config
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

logging.basicConfig(level=logging.INFO)

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

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

    # Initialize tokenizer, model, and image processor
    app.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6")
    app.model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM-V-2_6").to('cuda')
    app.image_processor = AutoImageProcessor.from_pretrained("openbmb/MiniCPM-V-2_6")

    return app