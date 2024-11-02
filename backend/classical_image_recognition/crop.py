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

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw contours on
contour_img = np.zeros(img.shape, dtype=np.uint8)

# Draw all contours on the blank image
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Display the original image and the image with contours
cv2.imshow('Original Image', img)
cv2.imshow('Contours', contour_img)

# Wait for a key press
cv2.waitKey(0)

# Close all windows

# print(contours)

# for contour in contours:
#     # Approximate the contour to a polygon
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)
    
#     # Check if the polygon has 4 vertices (rectangle)
#     if len(approx) == 4:
#         # Get the bounding rectangle
#         x, y, w, h = cv2.boundingRect(approx)
        
#         # Draw the rectangle on the original image
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         print(f"Rectangle found at: ({x}, {y}), width: {w}, height: {h}")

# # Display the image with detected rectangles
# cv2.imshow('Detected Rectangles', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Or save the image
# cv2.imwrite('detected_rectangles.png', img)