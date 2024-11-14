import cv2
import os

# Initialize camera
cap = cv2.VideoCapture(0)
user_name = input("Enter your name: ")
path = f'dataset/{user_name}'
os.makedirs(path, exist_ok=True)

count = 0
while count < 50:  # Capture 50 images for each user
    ret, frame = cap.read()
    if not ret:
        break
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y + h, x:x + w]
        cv2.imwrite(f"{path}/{user_name}_{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Face Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
