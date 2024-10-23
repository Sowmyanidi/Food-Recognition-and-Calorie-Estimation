import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import json

# Load the calorie dataset
calorie_df = pd.read_csv('calorie_dataset.csv')

# Load the model
model = tf.keras.models.load_model('classify.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

def predict_and_display_calories(img_path, model, calorie_df, class_indices):
    # Load and display the image
    img = Image.open(img_path)
    img = img.resize((224, 224))
    
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # Reverse the class indices to get class labels
    label_map = {v: k for k, v in class_indices.items()}
    predicted_label = label_map[predicted_class_index]
    
    # Fetch calorie information
    calorie_info = calorie_df[calorie_df['Label'] == predicted_label]
    
    if not calorie_info.empty:
        # Extract the information
        calorie_info_row = calorie_info.iloc[0]
        caloric_value = calorie_info_row['Caloric Value']
        carbohydrates = calorie_info_row['Carbohydrates']
        proteins = calorie_info_row['Proteins']
        fats = calorie_info_row['Fats']
        weight = calorie_info_row['Weight']
        
        print(f'Predicted label: {predicted_label}')
        print(f'There are {caloric_value} in {weight} of {predicted_label}.')
        
        # Pie chart for macronutrients
        labels = ['Carbohydrates', 'Proteins', 'Fats']
        sizes = [float(carbohydrates.split()[0]), float(proteins.split()[0]), float(fats.split()[0])]
        colors = ['#ff9999','#66b3ff','#99ff99']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title(f'Macronutrient Distribution of {predicted_label}')
        plt.show()
        
    else:
        print(f'No calorie information found for label: {predicted_label}')

# Example usage
img_path = 'b.png'  # Replace with your local test image path
predict_and_display_calories(img_path, model, calorie_df, class_indices)