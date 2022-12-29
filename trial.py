import cv2
import os
import numpy as np
import pandas as pd

#Import data of students from csv file
Record = pd.read_csv("./data.csv")

# Initialize variables
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
DIR = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys\Faces\train'
PWD = r'C:\Users\Asus\Desktop\BITS Books\CV\project\AttSys'

#All students who can register must have a folder in Faces/train
people = np.array(next(os.walk(DIR))[1])        
# print(people)

Registered = np.array(Record['Registered'])
# label = input('Enter Name: ')

label = 'Ishan'
valid = np.where(people == label)
# print(valid[0])
if(valid[0].size == 0):
    print("No such student")
    exit(0)

if(Registered[valid]):
    print("Already Registered")
    exit(0)

Record.loc[valid[0], 'Registered'] = 1

# writing into the file
Record.to_csv("data.csv", index=False)
