import cv2
import time
import os
import numpy as np
import pandas as pd
import imutils
import requests

#Import data of students from csv file
Record = pd.read_csv("./data.csv")

url = "http://192.168.116.31:8080/shot.jpg"

# Initialize variables
face_cascade = cv2.CascadeClassifier('./haar_face.xml')
DIR = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys\Faces\train'
PWD = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys'

# Initialize the webcam
# cap = cv2.VideoCapture(0)

# Initialize the list of faces and labels
faces = []
labels = []

#All students who can register must have a folder in Faces/train
people = np.array(next(os.walk(DIR))[1])        
# print(people)

Registered = np.array(Record['Registered'])
label = input('Enter Name: ')

# label = 'Alankar'
valid = np.where(people == label)
# print(valid[0])
if(valid[0].size == 0):
    print("No such student")
    exit(0)

if(Registered[valid]):
    print("Already Registered")
    exit(0)

path = os.path.join(DIR, label)
# os.makedirs(path)
os.chdir(path)
# exit(0)
i=0

while i in range(100):
    # Read a frame from the webcam
    # ret, frame = cap.read()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the img
    detected_faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    # Add the detected faces to the list of faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        imgName = str(i)+'.jpeg'
        i=i+1
        cv2.imwrite(imgName, gray)
        time.sleep(0.1)

    # Display the img
    cv2.imshow('Webcam', img)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27: # Press 'ESC' to quit
        exit(0)

# Release the webcam and destroy all windows
cv2.destroyAllWindows()

Record.loc[valid[0], 'Registered'] = 1

# writing into the file
Record.to_csv("data.csv", index=False)
