import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define the folder where HTML files are located
app = Flask(__name__, template_folder='frontend', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
MODEL_PATH = 'model/resnet50_synapsescan.h5'
model = load_model(MODEL_PATH)

# Define your class names (change if needed)
class_labels = ['Class_A', 'Class_B']  # Or HGSC, CCC, etc.

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route: About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Route: Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='❌ No image uploaded!')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='❌ No file selected.')

    # Save uploaded file to uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))  # match your model input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_labels[predicted_index]

    # Format output
    result = f"Predicted Class: {predicted_class}<br>📊 Confidence: {confidence:.2f}%"
    return render_template('index.html', prediction=result)

# ▶️ Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
