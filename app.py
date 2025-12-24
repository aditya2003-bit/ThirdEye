import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv8 model (small version for speed)
# It will download 'yolov8n.pt' automatically on first run
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Run inference
    results = model(img)

    # Process results to get object names
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            detections.append(object_name)

    # Create a natural language summary
    if not detections:
        summary = "I cannot see anything recognizable clearly."
    else:
        # Count objects (e.g., "2 chairs, 1 person")
        counts = {i: detections.count(i) for i in detections}
        summary_parts = [f"{count} {name}" for name, count in counts.items()]
        summary = "I see " + ", ".join(summary_parts)

    return jsonify({'message': summary})

if __name__ == '__main__':
    # Render requires the app to listen on 0.0.0.0 and a dynamic port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)