from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import gdown
import os

app = Flask(__name__)
CORS(app)

# Google Drive file ID and destination path
file_id = '1mb7q3DkULgPo3_z8xKPqEvtahAljw41A'  # Replace with your file ID
destination = '/tmp/plant_disease_model.h5'  # Temporary file path where the model will be saved

# Function to download the model from Google Drive
def download_model(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Download the model if it doesn't exist locally
if not os.path.exists(destination):
    download_model(file_id, destination)

# Load your trained model
model = keras.models.load_model(destination)

# Class names mapping
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_info = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a disease caused by the fungus Venturia inaequalis.',
        'treatment': 'Use fungicides, practice proper sanitation, and plant resistant varieties.',
        'prevention': 'Remove and destroy fallen leaves and fruit, and prune trees to improve air circulation.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease caused by Botryosphaeria obtusa.',
        'treatment': 'Apply appropriate fungicides and remove infected fruit and branches.',
        'prevention': 'Avoid injuries to the tree, and maintain proper tree health through fertilization and watering.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar-apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae.',
        'treatment': 'Apply fungicides and remove nearby juniper or cedar trees.',
        'prevention': 'Plant resistant varieties and remove galls from juniper or cedar trees.'
    },
    'Apple___healthy': {
        'description': 'Healthy apple plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Blueberry___healthy': {
        'description': 'Healthy blueberry plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew is a fungal disease that affects cherry trees, caused by Podosphaera clandestina.',
        'treatment': 'Use fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and avoid overhead irrigation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'Healthy cherry plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Gray leaf spot is a fungal disease of maize caused by Cercospora zeae-maydis.',
        'treatment': 'Apply fungicides and use resistant hybrids.',
        'prevention': 'Rotate crops and avoid overhead irrigation.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust is a fungal disease caused by Puccinia sorghi.',
        'treatment': 'Apply fungicides and use resistant hybrids.',
        'prevention': 'Plant resistant varieties and rotate crops.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is a fungal disease caused by Setosphaeria turcica.',
        'treatment': 'Apply fungicides and use resistant hybrids.',
        'prevention': 'Rotate crops and remove crop debris.'
    },
    'Corn_(maize)___healthy': {
        'description': 'Healthy corn plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot is a fungal disease caused by Guignardia bidwellii.',
        'treatment': 'Apply fungicides and remove infected fruit and leaves.',
        'prevention': 'Prune vines to improve air circulation and remove crop debris.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca, also known as Black Measles, is a complex fungal disease.',
        'treatment': 'Remove and destroy infected wood and apply fungicides.',
        'prevention': 'Avoid pruning during wet weather and disinfect pruning tools.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight is caused by the fungus Pseudopezicula tetraspora.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and avoid overhead irrigation.'
    },
    'Grape___healthy': {
        'description': 'Healthy grape plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Citrus greening is a bacterial disease caused by Candidatus Liberibacter species.',
        'treatment': 'No cure, remove infected trees to prevent spread.',
        'prevention': 'Control the Asian citrus psyllid vector and use disease-free planting material.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas campestris pv. pruni.',
        'treatment': 'Apply bactericides and remove infected leaves.',
        'prevention': 'Use resistant varieties and avoid overhead irrigation.'
    },
    'Peach___healthy': {
        'description': 'Healthy peach plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas campestris pv. vesicatoria.',
        'treatment': 'Apply bactericides and remove infected leaves.',
        'prevention': 'Use resistant varieties and avoid overhead irrigation.'
    },
    'Pepper,_bell___healthy': {
        'description': 'Healthy bell pepper plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight is a fungal disease caused by Alternaria solani.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Rotate crops and use resistant varieties.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight is a fungal disease caused by Phytophthora infestans.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Use resistant varieties and avoid overhead irrigation.'
    },
    'Potato___healthy': {
        'description': 'Healthy potato plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Raspberry___healthy': {
        'description': 'Healthy raspberry plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Soybean___healthy': {
        'description': 'Healthy soybean plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew is a fungal disease caused by Erysiphe cichoracearum and Sphaerotheca fuliginea.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and avoid overhead irrigation.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Leaf scorch of strawberry is caused by the fungus Diplocarpon earliana.',
        'treatment': 'Use fungicides and remove infected leaves.',
        'prevention': 'Plant resistant varieties, ensure good air circulation, and avoid overhead irrigation.'
    },
    'Strawberry___healthy': {
        'description': 'Healthy strawberry plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas species.',
        'treatment': 'Apply bactericides and remove infected leaves.',
        'prevention': 'Use resistant varieties and avoid overhead irrigation.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight is a fungal disease caused by Alternaria solani.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Rotate crops and use resistant varieties.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight is a fungal disease caused by Phytophthora infestans.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Use resistant varieties and avoid overhead irrigation.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold is a fungal disease caused by Fulvia fulva.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and avoid overhead irrigation.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot is a fungal disease caused by Septoria lycopersici.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Rotate crops and```python'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Two-spotted spider mite is an arachnid pest that feeds on tomato plants.',
        'treatment': 'Use miticides and release predatory mites.',
        'prevention': 'Maintain proper irrigation and avoid water stress.'
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot is a fungal disease caused by Corynespora cassiicola.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and rotate crops.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Tomato yellow leaf curl virus is a viral disease transmitted by whiteflies.',
        'treatment': 'Use insecticides to control whiteflies and remove infected plants.',
        'prevention': 'Use resistant varieties and practice good sanitation.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Tomato mosaic virus is a viral disease that affects tomato plants.',
        'treatment': 'Remove infected plants and control aphid vectors.',
        'prevention': 'Use virus-free seeds and resistant varieties.'
    },
    'Tomato___healthy': {
        'description': 'Healthy tomato plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    }
}

def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert the image to an array
    image = np.array(image) / 255.0
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = class_names[predicted_class]
    disease_details = disease_info[predicted_disease]
    return jsonify({
        'prediction': predicted_disease,
        'description': disease_details['description'],
        'treatment': disease_details['treatment'],
        'prevention': disease_details['prevention']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

