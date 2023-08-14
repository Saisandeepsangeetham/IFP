import os 
import cv2 as cv

#recording the video
cam = cv.VideoCapture(0)

cam.set(3,720)
cam.set(4,720)

face_detector = cv.CascadeClassifier("Face_detect/haar_cascade.xml")

face_id = input("Enter the face id:")

'''while True:
    face_id = input("Enter the face id: ")

    # Check if the entered ID is already taken
    if os.path.exists(f"dataset/user.{face_id}.1.jpg"):
        print("ID already exists. Please choose another ID.")
        continue  # Prompt for ID again
    else:
        break '''

count = 0

while (True):
    isTrue, img = cam.read()
    #for testing
    print("camera read:",isTrue)
    
    
    # camera frame is read...
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y),(x+w,y+h), (255,0,0), 2)
        count+=1
        
        cv.imwrite("Face_detect/dataset/user."+ str(face_id) + '.' + str(count) + ".jpg",img) 
    cv.imshow('sample', img)
        
    k = cv.waitKey(100) & 0xff
    
    if k == 27:
        break
    elif count >=55:
        break
cam.release()
cv.destroyAllWindows()