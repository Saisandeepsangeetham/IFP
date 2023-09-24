import cv2 as cv
import numpy as np

detector = cv.CascadeClassifier(r"Face_Detect/haar_cascade.xml")

recognizer = cv.face.LBPHFaceRecognizer.create()

recognizer.read("Face_Detect/face_trainer.yml")

img_path = "D:/IFP/Images/test.jpg"

names = ['None', 'Sai', 'Dhivagar']

frame = cv.imread(img_path)
    
if frame is None:
    print(f"Error: Unable to load the image from {img_path}")

else:
    max_width = 800  # Set the maximum width for the resized image

    height, width, _ = frame.shape
    if width > max_width:
        new_width = max_width
        new_height = int(height * (max_width / width))
        frame = cv.resize(frame, (new_width, new_height))
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

faces = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5)

for (x,y,w,h) in faces:
    face_roi = gray[y:y+h,x:x+w]
    
    id_,conf = recognizer.predict(face_roi)
    
    if conf <=100:
        font = cv.FONT_HERSHEY_SIMPLEX
        name = names[id_]
        
    else:
        font = cv.FONT_HERSHEY_SIMPLEX
        name = "unknown"
    
    cv.putText(frame,name,(x,y),font,1,(255,0,0),2)
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
    
cv.imshow("preview",frame)

cv.waitKey(0)
cv.destroyAllWindows()
