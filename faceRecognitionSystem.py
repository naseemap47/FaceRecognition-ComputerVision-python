import cv2
import os
from my_utils import findEncodings
import face_recognition as fr
import numpy as np

############################################
# Parameters
width = 640
height = 640
train_path = 'train'
test_path = 'test'
############################################

classNames = []
images = []
myList = os.listdir(train_path)
for cls in myList:
    classNames.append(cls)
    img = cv2.imread(f'{train_path}/{cls}')
    img = cv2.resize(img, (width, height))
    images.append(img)
# print(classNames)

# Train Encodings
train_encodings = findEncodings(images)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (width, height))  # Resize
    img_rgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # BGR to RGB
    face_locations = fr.face_locations(img_rgb)
    face_encodes = fr.face_encodings(img_rgb, face_locations)

    for encode, loc in zip(train_encodings, face_locations):
        matches = fr.compare_faces(train_encodings, encode)
        face_dist = fr.face_distance(train_encodings, encode)
        print(face_dist)

    cv2.imshow('Web-cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
