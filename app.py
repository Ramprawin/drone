from flask import Flask, render_template, request, Response
import cv2
import tensorflow as tf
import torch
import numpy as np
import winsound
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the plant health model
plant_health_model = load_model(r"C:\Users\rampr\Desktop\Animal\plant_health_model.h5")

# Define class labels for plant health analysis
plant_class_labels = [
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Target_Spot", 
    "Spider_mites Two-spotted_spider_mite", "Septoria_leaf_spot", "Leaf_Mold", 
    "Late_blight", "healthy", "Early_blight", "Bacterial_spot", "Leaf_scorch", 
    "Strawberry___healthy", "Squash___Powdery_mildew", "Soybean___healthy", 
    "Raspberry___healthy", "Potato___Late_blight", "Potato___healthy", 
    "Potato___Early_blight", "Pepper,_bell___healthy", "Pepper,_bell___Bacterial_spot", 
    "Peach___healthy", "Peach___Bacterial_spot", "Orange___Haunglongbing_(Citrus_greening)", 
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
    "Grape___Esca_(Black_Measles)", "Grape___Black_rot", "Corn_(maize)___Northern_Leaf_Blight", 
    "Corn_(maize)___healthy", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", 
    "Blueberry___healthy", "Apple___healthy", "Apple___Cedar_apple_rust", 
    "Apple___Black_rot", "Apple___Apple_scab"
]

# Load YOLO model for animal detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [14, 18, 19, 20]  # Animal classes

# Global variable to track mode
mode = "animal_detection"  # Switch between "animal_detection" and "plant_health"

def process_frame(frame):
    global mode
    if mode == "animal_detection":
        # Perform animal detection
        results = yolo_model(frame)
        detected_animals = results.pandas().xyxy[0]
        for index, row in detected_animals.iterrows():
            x1, y1, x2, y2, confidence, _, name = row
            if confidence > 0.5:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Play beep sound
                winsound.PlaySound(r"C:\Users\rampr\Desktop\Animal\static\beep-01a.wav", winsound.SND_ASYNC)
    elif mode == "plant_health":
        # Perform plant health analysis
        img = cv2.resize(frame, (150, 150)) / 255.0
        img_array = np.expand_dims(img, axis=0)
        predictions = plant_health_model.predict(img_array)
        predicted_class_idx = np.argmax(predictions)
        health_status = plant_class_labels[predicted_class_idx] if predictions[0][predicted_class_idx] > 0.5 else "Unknown"
        color = (0, 255, 0) if health_status == "healthy" else (0, 0, 255)
        cv2.putText(frame, f"Plant Health: {health_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

@app.route('/video_feed')
def video_feed():
    # Stream video frames from the drone camera
    def generate():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global mode
    mode = request.form.get('mode', 'animal_detection')
    return f"Switched to {mode}"

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
