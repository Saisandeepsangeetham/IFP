import cv2 as cv

#reading the image

img = cv.imread("Images\images.jpeg")

cv.imshow(img)
cv.waitKey(0)
print("hello world")