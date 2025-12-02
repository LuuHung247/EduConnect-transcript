from flask import Flask
from dotenv import load_dotenv
load_dotenv()

def create_app():
    app = Flask(__name__)
    from app.routes import bp as media_bp
    app.register_blueprint(media_bp)
    return app
