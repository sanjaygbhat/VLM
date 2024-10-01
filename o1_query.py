import os
from flask import Flask, render_template_string, request, jsonify
import requests
from openai import OpenAI
import os

app = Flask(__name__)


API_KEY = "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"
# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)



# HTML template with Bootstrap for styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT Query App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 50px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">GPT Query App</h1>
        <form method="post" id="queryForm">
            <div class="mb-3">
                <label for="query" class="form-label">Enter your query:</label>
                <textarea class="form-control" id="query" name="query" rows="3" required></textarea>
            </div>
            <button type="submit" class="btn btn-primary">Submit</button>
        </form>
        <div id="result" class="mt-4"></div>
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#queryForm').on('submit', function(e) {
                e.preventDefault();
                $.ajax({
                    url: '/query',
                    method: 'POST',
                    data: $(this).serialize(),
                    beforeSend: function() {
                        $('#result').html('<div class="alert alert-info">Processing your query...</div>');
                    },
                    success: function(response) {
                        $('#result').html('<div class="alert alert-success">' + response.result + '</div>');
                    },
                    error: function(xhr) {
                        $('#result').html('<div class="alert alert-danger">Error: ' + xhr.responseJSON.error + '</div>');
                    }
                });
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "o1-mini",  # OpenAI doesn't have an "o1" model, using gpt-3.5-turbo instead
                "messages": [{"role": "user", "content": user_query}]
            }
        )
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
        return jsonify({"result": result})
    except requests.RequestException as e:
        error_message = f"API request failed: {str(e)}"
        app.logger.error(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6555, debug=True)