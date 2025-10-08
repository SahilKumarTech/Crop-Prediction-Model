# import numpy as np
# # from flask import Flask,request,render_template
# # import pickle

# # flask_app = Flask(__name__)
# # model = pickle.load(open("crop_model.pkl","rb"))

# # @flask_app.route("/")
# # def home():
# #     return render_template("index.html")
# # @flask_app.route("/predict",methods=["POST"])
# # def predict():
# #     float_feature=[float(x) for x in request.form.values()]
# #     feature = [np.array(float_feature)]
# #     prediction = model.predict(feature)
# #     return render_template("index.html",prediction_text="The Prediction Crop is {}".format(prediction))
# # if __name__ == "__main__":
# #     flask_app.run(debug=True)

# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# import os

# app = Flask(__name__)

# # Load model with error handling
# try:
#     model = pickle.load(open("crop_model.pkl", "rb"))
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return render_template('index.html', prediction_text='Error: Model not loaded')
    
#     try:
#         # Get form values
#         n = float(request.form['N'])
#         p = float(request.form['P'])
#         k = float(request.form['K'])
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
        
#         # Create features array
#         features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
#         # Make prediction
#         prediction = model.predict(features)
#         return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')
    
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

# # Render specific configuration
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 10000))
#     app.run(host='0.0.0.0', port=port, debug=False)



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
