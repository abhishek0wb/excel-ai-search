import os
import pandas
from flask import Blueprint, request, jsonify
from Controller.api import handle_query

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route("/", methods=["POST"])
def vectorstore():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing or empty 'query' parameter"}), 400

    response = handle_query(query)
    return jsonify(response)
