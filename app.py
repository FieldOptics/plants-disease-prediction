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
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tea___white spot', 'Tea___Anthracnose',
    'Tea___healthy', 'Tea___bird eye spot', 'Tea___brown blight', 'Tea___red leaf spot',
    'Tea___gray light', 'Tea___algal leaf'
]

disease_info = {
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
    'Tea___white spot': {
        'description': 'White spot is a fungal disease caused by Pseudocercospora theae-sinensis, which results in small, circular, white spots on the leaves.',
        'treatment': 'Apply fungicides such as copper oxychloride and mancozeb to affected plants.',
        'prevention': 'Ensure good air circulation, avoid overhead irrigation, and remove infected plant debris.'
    },
    'Tea___Anthracnose': {
        'description': 'Anthracnose is caused by Colletotrichum spp., leading to dark, sunken lesions on leaves, stems, and fruits.',
        'treatment': 'Use fungicides containing copper or chlorothalonil and remove affected plant parts.',
        'prevention': 'Ensure proper spacing and air circulation, avoid overhead irrigation, and practice crop rotation.'
    },
    'Tea___healthy': {
        'description': 'Healthy tea plant with no visible signs of disease.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good horticultural practices.'
    },
    'Tea___bird eye spot': {
        'description': 'Bird eye spot, caused by Cercospora theae, produces small, round, brown spots with a white center, resembling a bird\'s eye.',
        'treatment': 'Apply fungicides like copper oxychloride and mancozeb.',
        'prevention': 'Improve air circulation, avoid overhead irrigation, and remove infected leaves.'
    },
    'Tea___brown blight': {
        'description': 'Brown blight, caused by Colletotrichum camelliae, leads to brown, irregular lesions on leaves and stems.',
        'treatment': 'Use fungicides such as copper oxychloride and mancozeb.',
        'prevention': 'Ensure proper spacing, improve air circulation, and avoid overhead irrigation.'
    },
    'Tea___red leaf spot': {
        'description': 'Red leaf spot, caused by Physalospora theicola, produces red to brown spots on leaves, leading to premature leaf drop.',
        'treatment': 'Apply fungicides like copper oxychloride and mancozeb.',
        'prevention': 'Enhance air circulation, avoid overhead irrigation, and remove infected plant debris.'
    },
    'Tea___gray light': {
        'description': 'Gray light, caused by Pestalotiopsis theae, results in grayish-white spots with dark margins on leaves.',
        'treatment': 'Use fungicides such as copper oxychloride and mancozeb.',
        'prevention': 'Improve air circulation, avoid overhead irrigation, and remove infected leaves.'
    },
    'Tea___algal leaf': {
        'description': 'Algal leaf spot, caused by Cephaleuros virescens, produces greenish-gray, velvety spots on leaves.',
        'treatment': 'Apply copper-based fungicides and prune affected areas to reduce humidity around the plants.',
        'prevention': 'Maintain good field hygiene, ensure adequate spacing between plants to improve air circulation, and avoid overhead irrigation.'
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
