import cv2
import face_recognition as fr


def findEncodings(images):
    face_encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encod = fr.face_encodings(img_rgb)[0]
        face_encodings.append(face_encod)
    return face_encodings
