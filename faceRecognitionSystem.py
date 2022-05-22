import cv2
import face_recognition as fr
import os

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
    images.append(img)
# print(classNames)

