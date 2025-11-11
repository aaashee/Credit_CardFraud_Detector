from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load

# Load the trained model
mod = load('creditcard.model')

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Prediction function
def predict_fraud(input_data):
    features = np.array(input_data).reshape(1, -1)
    prediction = mod.predict(features)
    return prediction[0]

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        input_data = [float(i) for i in data.split(',')]
        
        if len(input_data) != 30:
            return jsonify({'error': 'Please provide exactly 30 values.'}), 400
        
        prediction = predict_fraud(input_data)
        return jsonify({'prediction': int(prediction)})
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter numbers separated by commas.'}), 400

if __name__ == "__main__":
    app.run(debug=True)
