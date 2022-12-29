import cv2
import os
import pandas as pd
import numpy as np
import imutils
import requests

url = "http://192.168.116.31:8080/shot.jpg"
#Import data of students from csv file
Record = pd.read_csv("data.csv")

# Initialize variables
face_cascade = cv2.CascadeClassifier('./haar_face.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.xml')
people = next(os.walk('./Faces/train'))[1]
Students = np.array(Record['Name'])
Present = np.array(Record['Present'])
PWD = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys'
os.chdir(PWD)

# Initialize the webcam
# cap = cv2.VideoCapture(0)

c = 0
lp = len(people)+1
l = len(people)+1

while True:
    # Read a frame from the webcam
    # ret, frame = cap.read()

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    frame = imutils.resize(img, width=1000, height=1800)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detected_faces = face_cascade.detectMultiScale(gray, 1.8, 3)

    # Recognize the faces in the frame
    for (x, y, w, h) in detected_faces:
        label, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])

        # Check the confidence level
        if confidence < 8:
            # Display the label and confidence level
            print(f'{c} Label = {people[label]} with a confidence of {confidence}')

            cv2.putText(frame, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

            lp = label
            if l == lp:
                c = c+1
            else:
                c = 0
            
            if c > 10:
                Record.loc[label, 'Present'] = 1
                c = 0
                print("-----------------------")
            l = label

        else:
            # If the confidence is high, display 'Unknown'
            cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("Unknown")
            c = 0

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27: # Press 'ESC' to quit
        break

# Release the webcam and destroy all windows
# cap.release()
cv2.destroyAllWindows()

# writing into the file
Record.to_csv("data.csv", index=False)
