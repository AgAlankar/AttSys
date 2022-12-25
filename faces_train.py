import os
import cv2 as cv2
import numpy as np

# people = ['Ben Afflek', 'Ana de Armas', 'Robert Pattinson']
DIR = r'.\Faces\train'
people = next(os.walk(DIR))[1]

face_cascade = cv2.CascadeClassifier('./haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        print(path)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv2.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8)

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.xml')
# np.save('features.npy', features)
# np.save('labels.npy', labels)