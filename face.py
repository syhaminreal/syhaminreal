import cv2
import dlib
import numpy as np
from tkinter import Tk, filedialog
import os

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to detect face shape
def get_face_shape(landmarks):
    # Get the coordinates for the chin (jawline) points
    jaw_points = landmarks[0:17]
    jaw_width = np.linalg.norm(np.array(jaw_points[0]) - np.array(jaw_points[-1]))

    # Get the coordinates for forehead and chin
    forehead_point = landmarks[19]
    chin_point = landmarks[8]
    face_height = np.linalg.norm(np.array(forehead_point) - np.array(chin_point))

    # Analyze face dimensions (basic heuristic)
    ratio = face_height / jaw_width
    if ratio > 1.5:
        return "Oval"
    elif ratio > 1.3:
        return "Round"
    else:
        return "Square"

# Function to choose an image file using tkinter
def choose_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename()
    return file_path

# Function to process the image and detect face shape
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Loop through each face found
    for face in faces:
        # Get the landmarks
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # Detect the shape of the face
        face_shape = get_face_shape(landmarks_points)
        print(f"Detected face shape: {face_shape}")

        # Draw the landmarks on the image
        for point in landmarks_points:
            cv2.circle(img, point, 2, (0, 255, 0), -1)

    # Show the image with landmarks
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Select an image file
    image_file = choose_image_file()
    
    if os.path.exists(image_file):
        process_image(image_file)
    else:
        print("Image file not found.")
