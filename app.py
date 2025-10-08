import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

print("🚀 Starting Crop Prediction App...")

try:
    model = pickle.load(open("crop_model.pkl", "rb"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model loading failed')
    
    try:
        # Get all form values
        n = request.form.get('N', 0)
        p = request.form.get('P', 0) 
        k = request.form.get('K', 0)
        temp = request.form.get('temperature', 0)
        humidity = request.form.get('humidity', 0)
        ph = request.form.get('ph', 0)
        rainfall = request.form.get('rainfall', 0)
        
        # Convert to float
        features = [float(n), float(p), float(k), float(temp), float(humidity), float(ph), float(rainfall)]
        
        # Make prediction
        prediction = model.predict([features])
        crop_name = prediction[0]
        
        return render_template('index.html', prediction_text=f'Recommended Crop: {crop_name}')
        
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
