from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Run inference on a test image
results = model("test_image.jpg", conf=0.25)  # Adjust confidence threshold

# Display results
for result in results:
    result.show()
