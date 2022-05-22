import cv2
import face_recognition as fr

# Parameters
width = 640
height = 640

# Train
img1 = fr.load_image_file('train/Darshan.jpg')
img1 = cv2.resize(img1, (width, height))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(img1)
# print(faceLoc)
encode_img1 = fr.face_encodings(img1)
print(encode_img1)

# Test
img1T = fr.load_image_file('test/Darshan.jpg')
img1T = cv2.resize(img1T, (width, height))
img1T = cv2.cvtColor(img1T, cv2.COLOR_BGR2RGB)

faceLocT = fr.face_locations(img1T)
# print(faceLocT)
encode_img1T = fr.face_encodings(img1T)
print(encode_img1T)

cv2.imshow('Train', img1)
cv2.imshow('Test', img1T)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
