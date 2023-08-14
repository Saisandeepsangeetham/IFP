import os 
import cv2 as cv
import numpy as np
from PIL import Image

recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier("Face_detect/haar_cascade.xml")

dataset = os.g
