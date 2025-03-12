import os
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import concurrent.futures

app = Flask(__name__)

# Ensure the uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_symmetry(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image.", None

        # Resize image to a maximum width of 240 pixels
        height, width = image.shape[:2]
        if width > 240:
            scale = 240 / width
            image = cv2.resize(image, (240, int(height * scale)))

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect facial landmarks
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, "No face detected.", None

        # Extract landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        landmarks = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

        # Find midline points
        nose_tip = landmarks[1]  # Nose tip landmark
        mid_x = nose_tip[0]

        # Split left and right landmarks
        left_landmarks = landmarks[:234]
        right_landmarks = landmarks[234:468]

        # Mirror right landmarks for comparison
        right_landmarks_mirrored = np.array([[mid_x - (x - mid_x), y] for x, y in right_landmarks])

        # Calculate symmetry percentage
        differences = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
        symmetry_percentage = 100 - (np.mean(differences) / mid_x * 100)

        # Draw landmarks on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green dots for visualization

        # Save the annotated image
        annotated_image_path = os.path.join("uploads", "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)

        return symmetry_percentage, annotated_image_path, None
    except Exception as e:
        return None, str(e), None

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."})

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Please upload a PNG, JPG, or JPEG image."})

        # Ensure unique filename
        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # Calculate symmetry with timeout
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(calculate_symmetry, file_path)
                symmetry_percentage, annotated_image_path, error = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            return jsonify({"error": "Image processing timed out."})

        if error:
            return jsonify({"error": error})

        return jsonify({
            "symmetry_percentage": symmetry_percentage,
            "annotated_image": os.path.basename(annotated_image_path)
        })

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    app.run(debug=True)