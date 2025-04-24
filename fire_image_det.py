import tensorflow.lite as tflite
import numpy as np
import cv2
from PIL import Image

# Load TFLite model
interpreter = tflite.Interpreter(model_path="forest_fire_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the input size
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((img_width, img_height))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Load and preprocess test image
image_path = "C:/Users/naray/Desktop/fire_gan/archive/forest_fire/Training and Validation/fire/abc065.jpg"  # Replace with your image path
input_data = preprocess_image(image_path)

# Perform inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get prediction result
output_data = interpreter.get_tensor(output_details[0]['index'])
prediction = output_data[0][0]

# Interpret the result
if prediction > 0.01:
    print("🔥 Fire Detected!")
else:
    print("✅ No Fire Detected.")
