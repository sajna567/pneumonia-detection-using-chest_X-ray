from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained pneumonia classifier
model = load_model('models/vgg16_model.h5')
class_labels = ['Normal', 'Pneumonia','invalid']

# Upload folder
UPLOAD_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400

        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        label, confidence, scores, suggestion = predict_image(filepath)
        return render_template(
            'result.html',
            label=label,
            confidence=confidence,
            scores=scores,
            filename=file.filename,
            suggestion=suggestion
        )

    return render_template('upload.html')


def predict_image(img_path):
    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Predict using the trained model
    prediction = model.predict(img_array_expanded)[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    # Suggestion based on prediction
    if predicted_class == 'Pneumonia':
        suggestion = "Consult a physician for further evaluation and treatment."
    else:
        suggestion = "No signs of pneumonia detected. Continue routine checkups."

    return predicted_class, confidence, prediction, suggestion


if __name__ == '__main__':
    app.run(debug=True)