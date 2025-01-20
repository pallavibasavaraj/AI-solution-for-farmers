from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import os
import joblib




# Initialize Flask app
app = Flask(__name__)
app.secret_key = '1487a13ed87e0c939ce0d20a28512eb02f79389f0a43bb6a0ad702b3f65bb1e9'


# Load the trained model
model_path = "D:\project kisanmitr\models\Plant_Disease_Prediction_CNN_Image_Classifier.h5"  # Update this path if needed
plant_disease_model = load_model(model_path)



# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))  # Ensure this matches your model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_user():
    username = request.form['username']
    password = request.form['password']
    
    # Dummy login credentials (replace with secure authentication)
    users = {"admin": "password123"}
    if username in users and users[username] == password:
        session['user'] = username
        return redirect(url_for('home'))
    else:
        return "Invalid credentials! Please try again."

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')



@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"

        file = request.files['file']
        if file.filename == '':
            return "No file selected!"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image and predict
        image = preprocess_image(filepath)
        prediction = plant_disease_model.predict(image)
        disease_class = np.argmax(prediction)

        # Replace with your class labels
        class_labels = {0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}
        predicted_label = class_labels.get(disease_class, "Unknown")

         # Fertilizer recommendations
        fertilizer_recommendations = {
            'Apple___Apple_scab': "Use balanced NPK fertilizer with fungicide treatment.",
            'Apple___Black_rot': "Apply copper-based fungicide and balanced NPK fertilizer.",
            'Apple___Cedar_apple_rust': "Apply fungicides containing myclobutanil.",
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use fungicides and nitrogen-rich fertilizers.",
            'Corn_(maize)___Common_rust_': "Use nitrogen-rich fertilizer and fungicide.",
            'Corn_(maize)___Northern_Leaf_Blight': "Use phosphorus-rich fertilizers and fungicides.",
            'Potato___Early_blight': "Apply phosphorus-rich fertilizer with fungicide.",
            'Potato___Late_blight': "Use potassium-based fertilizer with metalaxyl-based fungicide.",
            'Tomato___Bacterial_spot': "Use copper-based fungicides and avoid overhead irrigation.",
            'Tomato___Early_blight': "Apply phosphorus-rich fertilizer and fungicides containing chlorothalonil.",
            'Tomato___Late_blight': "Use potassium-based fertilizer and fungicide.",
            'Tomato___Leaf_Mold': "Use balanced NPK fertilizer and apply fungicide sprays.",
            'Tomato___Septoria_leaf_spot': "Apply nitrogen-rich fertilizers and fungicide.",
            'Tomato___Target_Spot': "Use fungicides containing azoxystrobin or chlorothalonil.",
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whitefly population and use potassium-based fertilizer.",
            'Tomato___Tomato_mosaic_virus': "Ensure crop hygiene and use balanced fertilizers.",
            'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides and balanced fertilizers.",
            "Apple___Black_rot": "Apply copper-based fungicide and balanced NPK fertilizer.",
            "Apple___Cedar_apple_rust": "Apply fungicides containing myclobutanil.",
            "Apple___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Blueberry___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Cherry_(including_sour)___Powdery_mildew": "Use potassium-based fertilizer and apply fungicides like sulfur.",
            "Cherry_(including_sour)___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use fungicides and nitrogen-rich fertilizers.",
            "Corn_(maize)___Common_rust_": "Use nitrogen-rich fertilizer and fungicide.",
            "Corn_(maize)___Northern_Leaf_Blight": "Apply phosphorus-based fertilizer and fungicide sprays like azoxystrobin.",
            "Corn_(maize)___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Grape___Black_rot": "Apply balanced NPK fertilizer and fungicides like myclobutanil or captan.",
            "Grape___Esca_(Black_Measles)": "Use phosphorus-rich fertilizer and systemic fungicides.",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply potassium-rich fertilizer and fungicides containing copper.",
            "Grape___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Orange___Haunglongbing_(Citrus_greening)": "Use micronutrient-rich fertilizer containing magnesium and zinc, and ensure good pest control.",
            "Peach___Bacterial_spot": "Apply balanced NPK fertilizer and use copper-based bactericides.",
            "Peach___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Pepper,_bell___Bacterial_spot": "Apply phosphorus-rich fertilizer and bactericides like copper hydroxide.",
            "Pepper,_bell___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Potato___Early_blight": "Use potassium-rich fertilizer along with fungicides like mancozeb.",
            "Potato___Late_blight": "Apply phosphorus-based fertilizer and systemic fungicides such as metalaxyl.",
            "Potato___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Raspberry___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Soybean___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Squash___Powdery_mildew": "Apply potassium-based fertilizer and sulfur-based fungicides.",
            "Strawberry___Leaf_scorch": "Use nitrogen-rich fertilizer and fungicides like myclobutanil.",
            "Strawberry___healthy": "No fertilizer recommendation needed. Your crop is healthy!",
            "Tomato___Bacterial_spot": "Use copper-based bactericide and balanced NPK fertilizer.",
            "Tomato___Early_blight": "Apply phosphorus-rich fertilizer and fungicides like chlorothalonil.",
            "Tomato___Late_blight": "Use potassium-based fertilizer and fungicides like metalaxyl.",
            "Tomato___Leaf_Mold": "Apply nitrogen-rich fertilizer and fungicides such as mancozeb.",
            "Tomato___Septoria_leaf_spot": "Use balanced NPK fertilizer and apply fungicides containing copper or chlorothalonil.",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Apply potassium-rich fertilizer and use pest control solutions like neem oil.",
            "Tomato___Target_Spot": "Use phosphorus-rich fertilizer and fungicides like azoxystrobin.",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Apply micronutrient fertilizers (magnesium, zinc) and use insecticides to control whiteflies.",
            "Tomato___Tomato_mosaic_virus": "Use balanced NPK fertilizer and practice good sanitation and crop rotation.",
            "Tomato___healthy": "No fertilizer recommendation needed. Your crop is healthy!"







        }

        if "healthy" in predicted_label:
            recommended_fertilizer = "No recommendations needed; your crop is healthy."
        else:
            recommended_fertilizer = fertilizer_recommendations.get(predicted_label, "No specific recommendation available.")

        return render_template(
            'predict_disease.html',
            prediction=predicted_label,
            recommended_fertilizer=recommended_fertilizer,
            image_path=file.filename
        )



        return render_template('predict_disease.html', prediction=predicted_label, image_path=file.filename)

    return render_template('predict_disease.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True , port=5000)

