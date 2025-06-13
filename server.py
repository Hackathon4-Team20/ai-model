from flask import Flask, request, jsonify
from flasgger import Swagger
from satisfaction_tracker import SatisfactionTracker
import os
from dotenv import load_dotenv
import webbrowser
from threading import Timer

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)
swagger = Swagger(app)

# Initialize tracker with API key
tracker = SatisfactionTracker(openrouter_api_key=api_key.strip())

@app.route('/track', methods=['POST'])
def track_message():
    """
    Track customer satisfaction based on message.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            message:
              type: string
              example: "الخدمة ممتازة جداً"
    responses:
      200:
        description: Satisfaction tracking result
        schema:
          type: object
          properties:
            updated_score:
              type: integer
            status:
              type: string
            reason:
              type: string
    """
    data = request.get_json()
    message = data.get("message")
    
    # Default role to "user"
    result = tracker.add_message("user", message)
    
    return jsonify(result)

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, lambda: webbrowser.open('http://127.0.0.1:5000/apidocs')).start()
    app.run(debug=True)

