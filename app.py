import os
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

def calculate_symmetry(image_path):
    print(f"Processing image: {image_path}")  # Debug log
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to load image.")  # Debug log
            return None, "Failed to load image.", None

        # Resize image to a maximum width of 320 pixels
        height, width = image.shape[:2]
        if width > 320:
            scale = 320 / width
            image = cv2.resize(image, (320, int(height * scale)))

        print(f"Resized image dimensions: {image.shape}")  # Debug log

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Image converted to RGB.")  # Debug log

        # Process the image to detect facial landmarks
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            print("Error: No face detected.")  # Debug log
            return None, "No face detected.", None

        print("Face detected. Calculating symmetry...")  # Debug log

        # Extract landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape
        landmarks = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

        print(f"Number of landmarks: {len(landmarks)}")  # Debug log

        # Split landmarks into left and right
        mid_point = landmarks[1]  # Use the nose tip (landmark 1) as the midpoint
        left_landmarks = landmarks[:234]  # Left half of the face
        right_landmarks = landmarks[234:468]  # Right half of the face

        # Mirror right landmarks for comparison
        right_landmarks_mirrored = np.array([[mid_point[0] + (mid_point[0] - x), y] for x, y in right_landmarks])

        # Calculate symmetry percentage
        differences = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
        symmetry_percentage = 100 - (np.mean(differences) / mid_point[0] * 100)

        print(f"Symmetry percentage: {symmetry_percentage}")  # Debug log

        # Draw landmarks on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Smaller dots for better visualization

        # Save the annotated image
        annotated_image_path = os.path.join("uploads", "annotated_" + os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved at: {annotated_image_path}")  # Debug log

        # Release resources
        del image, image_rgb, results
        return symmetry_percentage, annotated_image_path, None
    except Exception as e:
        print(f"Error in calculate_symmetry: {e}")  # Debug log
        return None, str(e), None

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."})

        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Calculate symmetry and annotate image with a timeout
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(calculate_symmetry, file_path)
                symmetry_percentage, annotated_image_path, error = future.result(timeout=30)  # Timeout after 30 seconds
        except concurrent.futures.TimeoutError:
            print("Error: Image processing timed out.")  # Debug log
            return jsonify({"error": "Image processing timed out."})

        if error:
            return jsonify({"error": error})

        return jsonify({
            "symmetry_percentage": symmetry_percentage,
            "annotated_image": annotated_image_path
        })

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)