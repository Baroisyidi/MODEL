from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load YOLOv8 medium model for better accuracy
model = YOLO("yolov8m.pt")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template("index.html", error="Файл не выбран")
        
        file = request.files['file']
        
        # Check if file is valid
        if file.filename == '':
            return render_template("index.html", error="Файл не выбран")
        
        if not allowed_file(file.filename):
            return render_template("index.html", error="Недопустимый тип файла. Допустимы только JPG, JPEG, PNG.")

        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Preprocess image for better detection
            img = cv2.imread(upload_path)
            if img is None:
                raise ValueError("Не удалось прочитать загруженное изображение")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise
            img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)  # Enhance details

            # Run detection with optimized parameters
            results = model(
                img,
                conf=0.5,  # Higher confidence threshold
                classes=[41],  # Only detect cups (COCO class 41)
                imgsz=1280,  # Higher resolution for small objects
                iou=0.5  # Intersection over Union threshold
            )

            # Save annotated image - ensure directory exists
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            
            # Convert and save the image with quality settings
            success = cv2.imwrite(result_path, cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
            
            if not success:
                raise ValueError("Не удалось сохранить обработанное изображение")

            # Count detected cups
            cup_count = len(results[0].boxes)

            # Log results with additional metadata
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_filename": filename,
                "result_filename": result_filename,
                "cup_count": cup_count,
                "detections": [
                    {
                        "confidence": float(box.conf),
                        "position": box.xywh[0].tolist()
                    } for box in results[0].boxes
                ]
            }

            # Save to JSON log
            with open("data/logs/detections.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            return render_template(
                "result.html",
                original=url_for('static', filename=f'uploads/{filename}'),
                result=url_for('static', filename=f'results/{result_filename}'),
                count=cup_count,
                detections=log_entry["detections"],
                result_filename=result_filename  # Pass filename for debugging
            )

        except Exception as e:
            # Clean up failed uploads
            if 'upload_path' in locals() and os.path.exists(upload_path):
                os.remove(upload_path)
            return render_template("index.html", error=f"Изображение для обработки ошибок: {str(e)}")

    return render_template("index.html")

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("index.html", error="Слишком большой файл (не более 16 МБ)"), 413

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')