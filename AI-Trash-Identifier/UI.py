from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import io
from PIL import Image
import json
import requests

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file and prepare it for prediction
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict with the model
        prediction = model.predict(img_array)
        result = 1 if prediction[0] > 0.5 else 0

        # URL to make the POST request to
        url = 'https://a14b-198-96-33-206.ngrok-free.app/scan_results'

        # Headers for the POST request
        headers = {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        }

        # Create the payload with the prediction result
        payload = {
            'scan_result_code': result
        }

        # Make the POST request with the prediction result
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 201:
            print(f"POST successful: {response.json()}")
        else:
            print(f"POST failed: {response.status_code}, {response.text}")

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)