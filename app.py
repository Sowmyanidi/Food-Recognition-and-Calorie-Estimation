'''from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from PIL import Image
import json

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Updated path for static folder

# Load the model, dataset, and class indices
model = load_model('classify.h5')
calorie_df = pd.read_csv('calorie_dataset.csv')

# Load the class indices and reverse the mapping
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

label_map = {v: k for k, v in class_indices.items()}  # Reverse class indices

def predict_calories(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class of the food
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = label_map.get(predicted_class_index, "Unknown")

    # Fetch calorie information from the CSV
    calorie_info = calorie_df[calorie_df['Label'] == predicted_label]
    if not calorie_info.empty:
        row = calorie_info.iloc[0]
        return {
            'label': predicted_label,
            'caloric_value': row['Caloric Value'],
            'carbohydrates': row['Carbohydrates'],
            'proteins': row['Proteins'],
            'fats': row['Fats'],
            'weight': row['Weight']
        }
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Predict the food item and its calories
            result = predict_calories(filepath)
            if result:
                # Pass the result to the result.html template
                return render_template('result.html', result=result)
            else:
                return 'No calorie information found.'
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    image_file = request.files['image_file']
    if image_file:
        # Save the captured image file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(img_path)

        # Predict the food item and its calories
        result = predict_calories(img_path)
        if result:
            return render_template('result.html', result=result)
        else:
            return 'No calorie information found.'
    return 'No image file provided.'

if __name__ == '__main__':
    app.run(debug=True)'''


from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model and dataset
model = load_model('classify.h5')
calorie_data = pd.read_csv('calorie_dataset.csv')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Map index to food label
index_to_label = {v: k for k, v in class_indices.items()}

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def predict_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return index_to_label[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            food_label = predict_food(filepath)
            return redirect(url_for('result', food_label=food_label))
    return render_template('index.html')

@app.route('/result')
def result():
    food_label = request.args.get('food_label')
    food_info = calorie_data[calorie_data['Label'] == food_label].iloc[0]
    
    # Extract numeric values only and convert to float for decimals
    food_data = {
        'label': food_label,
        'calories': food_info['Caloric Value'].split()[0],
        'carbs': float(food_info['Carbohydrates'].split()[0]),  # Use float() here
        'proteins': float(food_info['Proteins'].split()[0]),    # Use float() here
        'fats': float(food_info['Fats'].split()[0])             # Use float() here
    }

    return render_template('result.html', food_data=food_data)

if __name__ == '__main__':
    app.run(debug=True)


