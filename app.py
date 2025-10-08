import os
from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

print("🌱 Starting Crop Prediction App...")

# Load the ML model
try:
    with open('crop_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text='Model not available')
    
    try:
        # Get form data
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        
        # Make prediction
        prediction = model.predict([data])
        result = prediction[0]
        
        return render_template('index.html', prediction_text=f'Recommended Crop: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: Please check your inputs')

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
