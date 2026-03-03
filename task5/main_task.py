import cv2
import os
import numpy as np

# Path to dataset
dataset_path = "dataset"

faces = []
labels = []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(int(label))

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

print("Training Completed ✅")

# Test image
test_image = cv2.imread("test.jpeg")
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in detected_faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (faces[0].shape[1], faces[0].shape[0]))

    label, confidence = recognizer.predict(face)

    print ("confidence:",confidence)
    

    if confidence <80 and label == 1:
        name = "Alia Bhatt"
    else:
        name = "Unknown"

    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(test_image, name, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Recognition", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

