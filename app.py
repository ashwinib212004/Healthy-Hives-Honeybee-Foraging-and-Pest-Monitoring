from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Load the YOLO model
model = YOLO("C:\\my_project\\best.pt")  # Use your model file

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file part in the request!", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected!", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Perform detection on the image
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"output_{file.filename}")
    frame = cv2.imread(file_path)

    results = model.predict(source=frame, conf=0.25, save=False)  # Confidence threshold
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class indices

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Resize image to fit within a specific dimension for display
    max_width, max_height = 800, 600
    height, width = frame.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))

    cv2.imwrite(output_path, frame)  # Save the processed image

    return render_template("index.html", result_image=f"/outputs/{os.path.basename(output_path)}")

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
