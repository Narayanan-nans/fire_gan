import tensorflow.lite as tflite
import numpy as np
import cv2

# Load TFLite model
interpreter = tflite.Interpreter(model_path="forest_fire_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the input size
input_shape = input_details[0]['shape']
img_height, img_width = input_shape[1], input_shape[2]

# Start video capture
cap = cv2.VideoCapture(0)  # Change to '1' or another index if you use an external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize the frame for model input
    input_frame = cv2.resize(frame, (img_width, img_height))
    input_data = np.expand_dims(input_frame / 255.0, axis=0).astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    # Display results
    if prediction > 0.01:
        label = "🔥 Fire Detected!"
        color = (0, 0, 255)
    else:
        label = "✅ No Fire"
        color = (0, 255, 0)

    # Show label on frame
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Live Fire Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
