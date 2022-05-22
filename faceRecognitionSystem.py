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
myList = os.listdir(train_path)
for cls in myList:
    classNames.append(cls)
print(classNames)
