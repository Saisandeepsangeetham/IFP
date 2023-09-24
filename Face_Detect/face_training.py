import os 
import cv2 as cv
import numpy as np
from PIL import Image

recognizer = cv.face.LBPHFaceRecognizer.create()
detector = cv.CascadeClassifier("Face_detect/haar_cascade.xml")

face_train = []
face_no = []

os.chdir("D:/IFP/Face_Detect/dataset")

data = os.getcwd()
#print(data)

if os.path.exists(os.getcwd()):
    for dirpath, dirname, filename in os.walk(os.getcwd()):
        for files in filename:
            #print(files)
            
            path = os.path.join(data,files)
            name_id = int(os.path.basename(path).split(".")[1])
            face_no.append(name_id) 
                       
            img = Image.open(path).convert("L")
            img_numpy = np.array(img,'uint8')
            
            face = detector.detectMultiScale(img_numpy, scaleFactor=1.1,minNeighbors=5)
            
            for (x,y,w,h) in face:
                face_roi = img_numpy[y:y+h, x:x+w]
                face_train.append(face_roi)
                #cv.rectangle(face,(x,y),(x+w,y+h),(255,0,0),2)
            #cv.imshow("sample",face)
else:
    print("The dataset directory is not found")

#print(face_no)

recognizer.train(face_train,np.array(face_no))
recognizer.save("D:/IFP/Face_Detect/face_trainer.yml")
