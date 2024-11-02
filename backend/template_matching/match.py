from file_to_img import pdf_to_images
import os

# path_to_pdf = '../testing_pdfs/L1.pdf'

folder_path = "../testing_pdfs/"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        pdf_to_images(file_path, 'images')
        
# pdf_to_images(path_to_pdf, 'images')

# import cv2
# import numpy as np

# image_path = 'images/page_1.png'
# template_path = 'templates/image194.png'

# image = cv2.imread(image_path)
# template = cv2.imread(template_path)

# def template_match(image, template, threshold=0.6):
#     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#     locations = np.where(result >= threshold)
#     return list(zip(*locations[::-1]))

# print(template_match(image, template))

import cv2
import numpy as np

image_path = 'images/page_1.png'
template_path = 'templates/Screenshot 2024-11-02 at 12.03.33.png'

def multi_scale_template_matching(image, template, scale_range=(0.2, 1.0, 0.05)):
    best_match = None
    best_score = -np.inf
    
    for scale in np.arange(scale_range[0], scale_range[1], scale_range[2]):
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        
        if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break
        
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = (max_loc, scale)
    
    return best_match

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def match_template_with_rotations(image, template, threshold=0.6):
    best_match = None
    best_score = -np.inf
    
    for angle in [0, 90, 180, 270]:
        rotated_template = rotate_image(template, angle)
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score and max_val >= threshold:
            best_score = max_val
            best_match = (max_loc, angle)
    
    return best_match

# Load image and template
image = cv2.imread(image_path, 0)
template = cv2.imread(template_path, 0)

# Perform matching
match = match_template_with_rotations(image, template)

if match:
    loc, angle = match
    h, w = template.shape
    top_left = loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, 255, 2)
    print(f"Best match found at {loc} with rotation {angle} degrees")
else:
    print("No match found above the threshold")

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
