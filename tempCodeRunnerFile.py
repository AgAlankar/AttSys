h)
# os.chdir(path)
# i=0

# while i in range(20):
#     # Read a frame from the webcam
#     ret, frame = cap.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the frame
#     detected_faces = face_cascade.detectMultiScale(gray, 1.5, 3)

#     # Add the detected faces to the list of faces
#     for (x, y, w, h) in detected_faces:
#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
#         imgName = str(i)+'.jpeg'
#         i=i+1
#         cv2.imwrite(imgName, gray)
#         time.sleep(0.05)

#     # Display the frame
#     cv2.imshow('Webcam', frame)

#     # Check for user input
#     key = cv2.waitKey(1)
#     if key == 27: # Press 'ESC' to quit
#         break