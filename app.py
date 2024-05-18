from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import io
import os

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to Load and Preprocess the Image using Pillow
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_file.read()))
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(image_file, model, class_indices):
    preprocessed_img = preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    print("Printing items")
    if request.method == 'POST':
        print("POST request received")
        if 'file' not in request.files:
            print("No file in request")
            return render_template('index.html', prediction="No image uploaded!")
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            return render_template('index.html', prediction="No image uploaded!")
        if file and allowed_file(file.filename):
            print("File uploaded and allowed")
            prediction = predict_image_class(file, model, class_indices)
            print(prediction)
            return prediction
    print("Returning render template without prediction")
    return render_template('index.html')



if __name__ == '__main__':
    app.run()
