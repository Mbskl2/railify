import cv2
import numpy as np
from backend.utils.file_to_img import pdf_to_images

# Setup pipline to read in Image and Extract Lines
def read_in_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

# Step 1 Crop

# Step 2 Crop


if __name__ == '__main__':
    image_path = 'backend/images/L1_4711_Tobel_page_1.png'
    img = read_in_image(image_path)
