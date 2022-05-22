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

# Train
encodings = findEncodings(images)
print(len(encodings))
