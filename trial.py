import cv2
import numpy as np

haar_cascade_f = cv2.CascadeClassifier('C:/Users/Asus/Desktop/BITS Books/CV/project/AttSys/haarcascade_frontalface_alt.xml')
haar_cascade = cv2.CascadeClassifier('C:/Users/Asus/Desktop/BITS Books/CV/project/AttSys/haar_face.xml')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces_f = haar_cascade_f.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=3)
    detected_faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=3)

    for (x, y, w, h) in detected_faces_f:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=2)

    cv2.imshow('Webcam', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27: # Press 'ESC' to quit
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()