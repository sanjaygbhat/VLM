from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import Config
import os

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

    # Ensure the upload and index directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

    with app.app_context():
        # Create database tables
        db.create_all()

    return app