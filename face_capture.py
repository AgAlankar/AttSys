import cv2
import time
import os
import numpy as np
# print(Record['Registered'])
# exit(0)

# Initialize variables
face_cascade = cv2.CascadeClassifier('./haar_face.xml')
DIR = r'.\Faces\train'

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the list of faces and labels
faces = []
labels = []

# label = input('Enter label: ')
label = 'Alankar'

path = os.path.join(DIR, label)
os.makedirs(path)
os.chdir(path)
i=0

while i in range(50):
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detected_faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    # Add the detected faces to the list of faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        imgName = str(i)+'.jpeg'
        # imagePath = os.path.join(path, imgName)
        # print(imagePath)
        i=i+1
        # cv2.imwrite(imagePath, gray[y:y+h, x:x+w])
        # cv2.imwrite('.', gray[y:y+h, x:x+w])
        # cv2.imwrite(imgName, gray[y:y+h, x:x+w])
        cv2.imwrite(imgName, gray)
        time.sleep(0.05)
        # cv2.imwrite('opencv'+str(i)+'.png', image)
        # faces.append(gray[y:y+h, x:x+w])

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27: # Press 'ESC' to quit
        break

# print(labels)
# print(faces)
# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()