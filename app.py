from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import dlib
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_symmetry(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        return None, "No face detected.", None

    # Get facial landmarks
    landmarks = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Split landmarks into left and right
    mid_point = landmarks[27]  # Tip of the nose (landmark 27)
    left_landmarks = landmarks[:27]
    right_landmarks = landmarks[27:54]

    # Mirror right landmarks for comparison
    right_landmarks_mirrored = np.array([[mid_point[0] + (mid_point[0] - x), y] for x, y in right_landmarks])

    # Calculate symmetry percentage
    differences = np.linalg.norm(left_landmarks - right_landmarks_mirrored, axis=1)
    symmetry_percentage = 100 - (np.mean(differences) / mid_point[0] * 100)

    # Draw landmarks on the image with larger dots
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Larger dots (radius = 5, filled)

    # Save the annotated image
    annotated_image_path = os.path.join("uploads", "annotated_" + os.path.basename(image_path))
    cv2.imwrite(annotated_image_path, image)
    print(f"Annotated image saved at: {annotated_image_path}")  # Debug print

    return symmetry_percentage, annotated_image_path, None

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."})

        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Calculate symmetry and annotate image
        symmetry_percentage, annotated_image_path, error = calculate_symmetry(file_path)
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