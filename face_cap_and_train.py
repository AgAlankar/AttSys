import cv2
import time
import os
import numpy as np
import pandas as pd

#Import data of students from csv file
Record = pd.read_csv("./data.csv")

# Initialize variables
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
DIR = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys\Faces\train'
PWD = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys'

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the list of faces and labels
faces = []
labels = []

#All students who can register must have a folder in Faces/train
people = np.array(next(os.walk(DIR))[1])        
# print(people)

Registered = np.array(Record['Registered'])
label = input('Enter label: ')

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
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detected_faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    # Add the detected faces to the list of faces
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        imgName = str(i)+'.jpeg'
        i=i+1
        cv2.imwrite(imgName, gray)
        time.sleep(0.1)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27: # Press 'ESC' to quit
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()

features = []
labels = []
# people.append(label)

os.chdir(DIR)

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        print(path)
        # label = people.index(person)
        label = np.where(people == person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv2.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=3)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label[0])

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

os.chdir(PWD)

face_recognizer.save('face_trained.xml')
# np.save('features.npy', features)
# np.save('labels.npy', labels)

Record.loc[valid[0], 'Registered'] = 1

# writing into the file
Record.to_csv("data.csv", index=False)
