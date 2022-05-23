import cv2
import face_recognition as fr
import datetime


def findEncodings(images):
    face_encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encod = fr.face_encodings(img_rgb)[0]
        face_encodings.append(face_encod)
    return face_encodings


def getAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        name_list = []
        for line in data:
            entry = line.split(',')
            name_list.append(entry[0])

        # if name NOT present in the list, it will add
        if name not in name_list:
            c_time = datetime.datetime.now()
            date_str = c_time.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_str}')
