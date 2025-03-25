from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Initialize Flask app with proper folders for templates and static files
app = Flask(__name__, static_url_path='/static', template_folder='templates')
CORS(app)

# Load the trained model (ensure plant_final_model.h5 is in your project root)
model = tf.keras.models.load_model('plant_final_model.h5')
class_names = [
    'Aloevera',
    'Amla', 
    'Amruta_Balli', 
    'Arali', 
    'Ashoka', 
    'Ashwagandha', 
    'Avacado', 
    'Bamboo', 
    'Basale', 
    'Betel', 
    'Betel_Nut',
    'Brahmi', 
    'Castor',
    'Curry_Leaf', 
    'Doddapatre', 
    'Ekka', 
    'Ganike', 
    'Gauva', 
    'Geranium', 
    'Henna', 
    'Hibiscus', 
    'Honge', 
    'Insulin',
    'Jasmine', 
    'Lemon', 
    'Lemon_grass', 
    'Mango', 
    'Mint', 
    'Nagadali', 
    'Neem', 
    'Nithyapushpa', 
    'Nooni', 
    'Pappaya', 
    'Pepper', 
    'Pomegranate', 
    'Raktachandini', 
    'Rose', 
    'Sapota', 
    'Tulasi', 
    'Wood_sorel',
]

# Route for the landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the homepage (after landing page)
@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

# Route for after capturing an image (showing the captured image)
@app.route('/afterclick')
def afterclick():
    return render_template('afterclick.html')

# Route for settings
@app.route('/setting')
def setting():
    return render_template('setting.html')

# Route for camera page (plant identification)
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Prediction endpoint for the image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        # Preprocess the image
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction using your model
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({
            'plant_type': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5000 with debugging enabled
    app.run(debug=True, port=5000)
