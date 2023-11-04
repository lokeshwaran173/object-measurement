import cv2 
import numpy as np 

# Load the image 
img = cv2.imread('snapshots/five.png') 

# Convert the image to grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

# Apply a threshold to the image to 
# separate the objects from the background 
ret, thresh = cv2.threshold( 
	gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

# Find the contours of the objects in the image 
contours, hierarchy = cv2.findContours( 
	thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

# Loop through the contours and calculate the area of each object 
for cnt in contours: 
	area = cv2.contourArea(cnt) 

	# Draw a bounding box around each 
	# object and display the area on the image 
	x, y, w, h = cv2.boundingRect(cnt) 
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
	cv2.putText(img, str(area), (x, y), 
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

# Show the final image with the bounding boxes 
# and areas of the objects overlaid on top 
cv2.imshow('image', img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

# Code By SR.Dhanush 
