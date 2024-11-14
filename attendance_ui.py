import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import datetime

# Load the pre-trained face recognition model
model = load_model('models/face_recognition_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.datetime.now()
        f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Function to capture and recognize faces
def recognize_face():
    classes = os.listdir('dataset')

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Streamlit interface elements
    st.title('Attendance Management System')
    st.subheader('Face Recognition Based Attendance')

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the face from the frame
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (128, 128))
            face = np.expand_dims(face, axis=0)

            # Predict the person from the face
            prediction = model.predict(face)
            class_id = np.argmax(prediction)
            name = classes[class_id]
            
            # Mark attendance for recognized person
            mark_attendance(name)

            # Draw a rectangle around the face and put the name
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame with face recognition
        st.image(frame, channels='BGR', use_column_width=True)

        # End the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Streamlit button to start face recognition
if st.button('Start Attendance'):
    st.write('Starting face recognition for attendance...')
    recognize_face()
