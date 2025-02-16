from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load YOLOv8 model
MODEL_PATH = "best.pt"  # Ensure best.pt is in the same directory
model = YOLO(MODEL_PATH)

# Class names from your yaml file
class_names = ['Fani 1', 'Fani2', 'Masjed', 'library']

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "YOLOv8 Flask API is running! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Read the image
    image = cv2.imread(filename)

    # Run YOLOv8 inference
    results = model(image)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes.data:  # Iterate over detected boxes
            # Check if the box has the correct number of elements
            if len(box.tolist()) >= 6:
                x1, y1, x2, y2, conf, cls = map(float, box.tolist()[:6])
                
                # Map class id to class name
                class_name = class_names[int(cls)] if int(cls) < len(class_names) else "Unknown"
                
                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(conf),
                    "class": class_name  # Using class name instead of id
                })

                # Draw bounding box on image (optional, for debugging or visualization)
                label = f"{class_name}: {conf:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the modified image with bounding boxes (optional, for debugging or visualization)
    output_filename = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(output_filename, image)

    # Return only the detections, without the image URL
    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
