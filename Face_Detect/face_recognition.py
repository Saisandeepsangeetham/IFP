import os 
import cv2 as cv
import numpy as np
import pyttsx3
from PIL import Image

detector = cv.CascadeClassifier(r"Face_Detect/haar_cascade.xml")

recognizer = cv.face.LBPHFaceRecognizer.create()
recognizer.read(r"Face_Detect/face_trainer.yml")

cam = cv.VideoCapture(0)

<<<<<<< HEAD
names = ['None','sai']

engine = pyttsx3.init()

engine.setProperty('rate', 150)

detected_faces = set()
=======
names = ['None','Sai','Nishi','Teja','Mom']
>>>>>>> Branch1

while True:
    isTrue, frame = cam.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        id_,conf = recognizer.predict(face_roi)
        font = cv.FONT_HERSHEY_SIMPLEX
        
        #print(conf)
        #print(id_)
        font = cv.FONT_HERSHEY_SIMPLEX
        
        if conf <=100:
            name = names[id_]
            
        else:
            name = "unknown"

        cv.putText(frame, name, (x,y),font, 1, (255,0,0), 2)
        cv.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), thickness = 2)
        
        if name not in detected_faces:
            engine.say(f"Detected face: {name}")
            engine.runAndWait()
            detected_faces.add(name)
            
    cv.imshow("preview",frame)
    
    k = cv.waitKey(10) & 0xff
    if k==27:
        break
    
cam.release()
cv.destroyAllWindows()
    
    