import cv2
import numpy as np
from modelhub.onnx import InsightFace2D106, InsightFaceSwap, YoloV5Face

def swap_faces(image_path1, image_path2, device='cpu'):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Initialize models
    face_detector = YoloV5Face(device)
    face_marker = InsightFace2D106(device)
    face_swapper = InsightFaceSwap(device)

    # Detect faces in both images
    faces1 = face_detector.extract(image1, threshold=0.5)[0]
    faces2 = face_detector.extract(image2, threshold=0.5)[0]

    # Ensure each image has at least one face detected
    if len(faces1) == 0 or len(faces2) == 0:
        print("Faces not detected in one or both images.")
        return None

    # Use the first detected face
    face1 = max(faces1, key=lambda x: (x[3]-x[1])*(x[2]-x[0]))  # largest face
    face2 = max(faces2, key=lambda x: (x[3]-x[1])*(x[2]-x[0]))

    # Extract landmarks for alignment
    landmarks1 = face_marker.extract(cv2.cvtColor(image1[face1[1]:face1[3], face1[0]:face1[2]], cv2.COLOR_BGR2RGB))[0]
    landmarks2 = face_marker.extract(cv2.cvtColor(image2[face2[1]:face2[3], face2[0]:face2[2]], cv2.COLOR_BGR2RGB))[0]

    # Generate face vectors
    face_vector1 = face_swapper.get_face_vector(image1, landmarks1)
    face_vector2 = face_swapper.get_face_vector(image2, landmarks2)

    # Swap faces
    swapped_face1 = face_swapper.generate(image1, face_vector2)
    swapped_face2 = face_swapper.generate(image2, face_vector1)

    # Place the swapped faces back to the original images
    image1[face1[1]:face1[3], face1[0]:face1[2]] = cv2.resize(swapped_face2, (face1[2]-face1[0], face1[3]-face1[1]))
    image2[face2[1]:face2[3], face2[0]:face2[2]] = cv2.resize(swapped_face1, (face2[2]-face2[0], face2[3]-face2[1]))

    return image1, image2

# Usage example:
result_image1, result_image2 = swap_faces('bradley.jpg', 'jim.jpg')
cv2.imshow('Swapped Image 1', result_image1)
cv2.imshow('Swapped Image 2', result_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
