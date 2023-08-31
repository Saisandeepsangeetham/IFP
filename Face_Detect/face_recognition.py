import os 
import cv2 as cv
import numpy as np
from PIL import Image

detector = cv.CascadeClassifier(r"Face_Detect/haar_cascade.xml")

recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read(r"Face_Detect/face_trainer.yml")

cam = cv.VideoCapture(0)

names = ['None','Sai','Dhivagar','Mom','super']

while True:
    isTrue, frame = cam.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        id_,conf = recognizer.predict(face_roi)
        #print(conf)
        #print(id_)
        
        if conf <=100:
            font = cv.FONT_HERSHEY_SIMPLEX
            name = names[id_]
            
        else:
            name = "unknown"

        cv.putText(frame, name, (x,y),font, 1, (255,0,0), 2)
        cv.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), thickness = 2)
        
    cv.imshow("preview",frame)
    
    k = cv.waitKey(10) & 0xff
    if k==27:
        break
    
cam.release()
cv.destroyAllWindows()
    
    