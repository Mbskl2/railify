#mark all the boxes at first
import cv2
import numpy as np
import os

print(os.getcwd())
# Read the image
img = cv2.imread('backend/images/L1_66666_Schoenwald_page_1.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print(contours)

for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if the polygon has 4 vertices (rectangle)
    if len(approx) == 4:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Draw the rectangle on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        print(f"Rectangle found at: ({x}, {y}), width: {w}, height: {h}")

# Display the image with detected rectangles
cv2.imshow('Detected Rectangles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Or save the image
cv2.imwrite('detected_rectangles.png', img)