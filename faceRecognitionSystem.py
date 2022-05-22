import cv2
import os
from my_utils import findEncodings

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
encodings = findEncodings(images)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    cv2.imshow('Web-cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
