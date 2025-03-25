import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('plant_final_model.h5')

# Ensure this variable exists for import
plat_model1= model

# Define soil classes
PLANT_CLASSES = [
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

def predict_plant(image_array):
    """Predicts plant type from an image array"""
    img = tf.image.resize(image_array, [256, 256])  # Normalize
    img = img / 255.0  # Resize for model
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return PLANT_CLASSES[prediction.argmax()]