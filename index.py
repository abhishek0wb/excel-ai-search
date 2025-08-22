from flask import Flask
from Route.routes import api_blueprint

# Initialize Flask app
app = Flask(__name__)

# Register the blueprint
app.register_blueprint(api_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
