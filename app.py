import os
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

print("🚀 Starting Crop Prediction App...")

# Load model
try:
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded properly')
    
    try:
        # Get form data
        n = float(request.form['N'])
        p = float(request.form['P'])
        k = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Create features array
        features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        # Predict
        prediction = model.predict(features)
        crop = prediction[0]
        
        return render_template('index.html', prediction_text=f'Recommended Crop: {crop}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
