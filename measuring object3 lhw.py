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

# Set a minimum area threshold for "big" objects
min_area_threshold = 1000  # Adjust this value as needed

# Loop through the contours and calculate the area, length, height, and width of each "big" object
for cnt in contours:
    area = cv2.contourArea(cnt)

    # Check if the area of the object is greater than the threshold
    if area > min_area_threshold:
        x, y, w, h = cv2.boundingRect(cnt)
        length = max(w, h)
        height = min(w, h)
        
        # Draw a bounding box around each "big" object and display the area, length, height, and width on the image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Area: {int(area)}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Length: {length}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Height: {height}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Width: {w}", (x, y + h + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the final image with the bounding boxes
# and the area, length, height, and width of the "big" objects overlaid on top
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
