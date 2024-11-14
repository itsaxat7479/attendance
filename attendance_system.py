import datetime
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model('models/face_recognition_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.datetime.now()
        f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")

classes = os.listdir('dataset')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face)
        class_id = np.argmax(prediction)
        name = classes[class_id]
        mark_attendance(name)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
