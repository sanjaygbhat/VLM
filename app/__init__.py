from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import Config
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

def create_app(rank=0, world_size=1, config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Initialize distributed setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Import models
    from app.models.user import User
    from app.models.document import Document

    # Import and register blueprints
    from app.routes import auth, document, query
    app.register_blueprint(auth.bp)
    app.register_blueprint(document.bp)
    app.register_blueprint(query.bp)

    return app