from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from config import Config
from datetime import timedelta

db = SQLAlchemy()
jwt = JWTManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # JWT configurations
    app.config['JWT_SECRET_KEY'] = app.config['SECRET_KEY']
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

    db.init_app(app)
    jwt.init_app(app)

    from app.routes import auth, document, query
    app.register_blueprint(auth.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(query.bp)

    return app

__all__ = ['create_app', 'db']