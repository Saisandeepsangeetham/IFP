import cv2 as cv

#reading the image

img = cv.imread("Images\images.jpeg")

cv.imshow('sample',img)

cv.waitKey(0)
cv.destroyAllWindows()