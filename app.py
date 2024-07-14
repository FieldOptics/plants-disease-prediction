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
    'Blueberry___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
   
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 
     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_info = {
      'Apple___Apple_scab': {
        'description': 'Apple scab is a disease caused by the fungus Venturia inaequalis. It affects both leaves and fruit, causing dark, sunken lesions on the surface. Infected leaves may drop prematurely, leading to reduced yields and fruit quality.',
        'treatment': 'Use fungicides at the first sign of infection. Practice proper sanitation by removing and destroying fallen leaves and infected fruit. Plant resistant apple varieties to reduce susceptibility.',
        'prevention': 'Prune trees to improve air circulation, which helps reduce the humidity that favors fungal growth. Regularly monitor trees for early signs of the disease, and apply protective fungicide sprays during susceptible periods.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease caused by Botryosphaeria obtusa. It affects the fruit, leaves, and branches of apple trees, causing dark, sunken lesions on the fruit and leaves, and cankers on the branches.',
        'treatment': 'Apply appropriate fungicides to control the spread of the disease. Remove and destroy infected fruit and branches. Prune out cankers and disinfect pruning tools between cuts.',
        'prevention': 'Avoid injuring the tree, as wounds can serve as entry points for the fungus. Maintain proper tree health through balanced fertilization and adequate watering. Regularly monitor for signs of the disease and take early action.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar-apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae. It requires two hosts to complete its life cycle: apple trees and juniper or cedar trees. The disease causes bright orange, gelatinous galls on juniper or cedar trees and yellow-orange spots on apple leaves.',
        'treatment': 'Apply fungicides to apple trees at the first sign of infection. Remove nearby juniper or cedar trees if possible, or prune out galls from these trees. Regularly monitor both host trees for signs of the disease.',
        'prevention': 'Plant resistant apple varieties to reduce susceptibility. Increase the distance between apple trees and juniper or cedar trees to minimize disease spread. Practice good sanitation by removing and destroying fallen leaves and infected fruit.'
    },
    'Apple___healthy': {
        'description': 'A healthy apple plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and fruit develops normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pruning. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
    'Blueberry___healthy': {
        'description': 'A healthy blueberry plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and fruit develops normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pruning. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
   
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Gray leaf spot is a fungal disease of maize caused by Cercospora zeae-maydis. It causes rectangular, grayish lesions on the leaves, which can coalesce and lead to extensive leaf blight. Severe infections can reduce photosynthetic capacity and yield.',
        'treatment': 'Apply fungicides at the first sign of infection. Use resistant hybrids to reduce susceptibility. Remove and destroy infected crop residues to reduce inoculum levels.',
        'prevention': 'Rotate crops to reduce the buildup of inoculum in the soil. Avoid overhead irrigation, which can increase humidity and promote fungal growth. Regularly monitor fields for early signs of the disease and take early action.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust is a fungal disease of maize caused by Puccinia sorghi. It produces small, reddish-brown pustules on the leaves, which can coalesce and cause extensive leaf blight. Severe infections can reduce photosynthetic capacity and yield.',
        'treatment': 'Apply fungicides at the first sign of infection. Use resistant hybrids to reduce susceptibility. Remove and destroy infected crop residues to reduce inoculum levels.',
        'prevention': 'Plant resistant varieties to reduce susceptibility. Rotate crops to reduce the buildup of inoculum in the soil. Regularly monitor fields for early signs of the disease and take early action.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is a fungal disease of maize caused by Setosphaeria turcica. It produces large, elongated, grayish lesions on the leaves, which can coalesce and cause extensive leaf blight. Severe infections can reduce photosynthetic capacity and yield.',
        'treatment': 'Apply fungicides at the first sign of infection. Use resistant hybrids to reduce susceptibility. Remove and destroy infected crop residues to reduce inoculum levels.',
        'prevention': 'Rotate crops to reduce the buildup of inoculum in the soil. Avoid overhead irrigation, which can increase humidity and promote fungal growth. Regularly monitor fields for early signs of the disease and take early action.'
    },
   'Corn_(maize)___healthy': {
        'description': 'A healthy corn plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and plants develop normally with no stunted growth.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pest management. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot is a fungal disease caused by Guignardia bidwellii. It affects the leaves, stems, and fruit of grapevines, causing dark, sunken lesions on the fruit and circular lesions on the leaves with dark borders.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves, fruit, and vines. Prune vines to improve air circulation, which helps reduce the humidity that favors fungal growth.',
        'prevention': 'Ensure good air circulation around vines by proper spacing and pruning. Avoid overhead irrigation, which can increase humidity. Plant resistant varieties to reduce susceptibility.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca, also known as Black Measles, is a complex fungal disease. It causes dark streaks and spots on the leaves and fruit, and can lead to sudden vine collapse in severe cases.',
        'treatment': 'Remove and destroy infected wood and apply fungicides. Avoid pruning during wet weather and disinfect pruning tools between cuts.',
        'prevention': 'Plant resistant varieties to reduce susceptibility. Maintain good horticultural practices and regularly monitor vines for early signs of the disease.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight is caused by the fungus Pseudopezicula tetraspora. It causes small, angular lesions on the leaves, which can coalesce and cause extensive leaf blight in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Prune vines to improve air circulation, which helps reduce the humidity that favors fungal growth.',
        'prevention': 'Ensure good air circulation around vines by proper spacing and pruning. Avoid overhead irrigation, which can increase humidity. Plant resistant varieties to reduce susceptibility.'
    },
    'Grape___healthy': {
        'description': 'A healthy grape plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and fruit develops normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pruning. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
   
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas campestris pv. vesicatoria. It causes small, water-soaked lesions on the leaves and fruit, which can coalesce and cause extensive damage in severe cases.',
        'treatment': 'Apply bactericides at the first sign of infection. Remove and destroy infected leaves and fruit. Prune plants to improve air circulation, which helps reduce the humidity that favors bacterial growth.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Pepper,_bell___healthy': {
        'description': 'A healthy bell pepper plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and fruit develops normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pruning. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It causes small, dark brown lesions on the leaves with concentric rings, and can lead to leaf blight in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Rotate crops to reduce the buildup of inoculum in the soil.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight is a fungal disease caused by Phytophthora infestans. It causes large, dark brown lesions on the leaves and stems, and can lead to rapid plant collapse in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves and stems. Rotate crops to reduce the buildup of inoculum in the soil.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Potato___healthy': {
        'description': 'A healthy potato plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and tubers develop normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pest management. Regularly inspect plants for signs of disease or pests and take early action if needed.'
    },
   
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas species. It causes small, water-soaked lesions on the leaves and fruit, which can coalesce and cause extensive damage in severe cases.',
        'treatment': 'Apply bactericides at the first sign of infection. Remove and destroy infected leaves and fruit. Prune plants to improve air circulation, which helps reduce the humidity that favors bacterial growth.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It causes small, dark brown lesions on the leaves with concentric rings, and can lead to leaf blight in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Rotate crops to reduce the buildup of inoculum in the soil.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight is a fungal disease caused by Phytophthora infestans. It causes large, dark brown lesions on the leaves and stems, and can lead to rapid plant collapse in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves and stems. Rotate crops to reduce the buildup of inoculum in the soil.',
        'prevention': 'Use resistant varieties to reduce susceptibility. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold is a fungal disease caused by Fulvia fulva. It causes yellow spots on the upper leaf surface and a moldy growth on the underside. Severe infections can lead to leaf drop and reduced yield.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Ensure good air circulation around plants by proper spacing and pruning.',
        'prevention': 'Avoid overhead irrigation, which can increase humidity. Plant resistant varieties to reduce susceptibility. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot is a fungal disease caused by Septoria lycopersici. It causes small, dark spots on the leaves, which can coalesce and cause extensive leaf blight in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Ensure good air circulation around plants by proper spacing and pruning.',
        'prevention': 'Rotate crops to reduce the buildup of inoculum in the soil. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Two-spotted spider mite is an arachnid pest that feeds on tomato plants. It causes yellowing and stippling of the leaves, and can lead to leaf drop and reduced yield in severe cases.',
        'treatment': 'Use miticides at the first sign of infestation. Release predatory mites to control the population. Remove and destroy heavily infested leaves.',
        'prevention': 'Maintain proper irrigation to avoid water stress, which can make plants more susceptible to spider mite infestations. Regularly monitor plants for early signs of infestation and take early action.'
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot is a fungal disease caused by Corynespora cassiicola. It causes small, dark spots on the leaves with concentric rings, and can lead to leaf blight in severe cases.',
        'treatment': 'Apply fungicides at the first sign of infection. Remove and destroy infected leaves. Ensure good air circulation around plants by proper spacing and pruning.',
        'prevention': 'Rotate crops to reduce the buildup of inoculum in the soil. Avoid overhead irrigation, which can increase humidity. Regularly monitor plants for early signs of the disease and take early action.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Tomato yellow leaf curl virus is a viral disease transmitted by whiteflies. It causes yellowing and curling of the leaves, stunted growth, and reduced yield. There is no cure for the disease.',
        'treatment': 'Use insecticides to control whiteflies and remove infected plants to prevent the spread of the virus.',
        'prevention': 'Plant resistant varieties to reduce susceptibility. Practice good sanitation by removing and destroying infected plants. Implement control measures for whiteflies, including insecticides and biological control agents.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Tomato mosaic virus is a viral disease that affects tomato plants. It causes mottling and distortion of the leaves, stunted growth, and reduced yield. There is no cure for the disease.',
        'treatment': 'Remove and destroy infected plants to prevent the spread of the virus. Control aphid vectors with insecticides.',
        'prevention': 'Use virus-free seeds and resistant varieties to reduce susceptibility. Practice good sanitation by removing and destroying infected plants. Implement control measures for aphid vectors, including insecticides and biological control agents.'
    },
    'Tomato___healthy': {
        'description': 'A healthy tomato plant shows no visible signs of disease. Leaves are green and free of spots or lesions, and fruit develops normally without blemishes.',
        'treatment': 'No treatment is needed for a healthy plant.',
        'prevention': 'Maintain good horticultural practices, including proper watering, fertilization, and pest management. Regularly inspect plants for signs of disease or pests and take early action if needed.'
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

