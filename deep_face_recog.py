import caer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler


# test_path = r'../input/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'

# img = cv.imread(test_path)

# plt.imshow(img)
# plt.show()

cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detected_faces = face_cascade.detectMultiScale(gray, 1.8, 3)

    # Recognize the faces in the frame
    for (x, y, w, h) in detected_faces:
        label, confidence = face_recognizer.predict(gray[y:y+h, x:x+w])

        # Check the confidence level
        if confidence < 40:
            # Display the label and confidence level
            faces_roi = gray[y:y+h,x:x+w]

            label, confidence = face_recognizer.predict(faces_roi)
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
cap.release()
cv2.destroyAllWindows()


def prepare(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

predictions = model.predict(prepare(img))

# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])