import numpy as np
from PIL import Image

def is_grey(pixel):
    return pixel < 250

def remove_grey_boxes(image_path, output_path, window_size=5, buffer_size=2):
    # Open the image
    img = Image.open(image_path)
    img_array = np.array(img)

    greyscale = np.mean(img_array, axis=2).astype(np.uint8)
    
    height, width, _ = img_array.shape
    
    # Create a copy of the image to modify
    result = np.copy(img_array)
    
    for y in range(0, height - window_size + 1, int(window_size)):
        for x in range(0, width - window_size + 1, int(window_size)):
            window = greyscale[y:y+window_size, x:x+window_size]
            
            # Check if all pixels in the window are grey
            if np.all([is_grey(pixel) for row in window for pixel in row]):
                # Calculate the area to turn white, including the buffer
                y_start = max(0, y - buffer_size)
                y_end = min(height, y + window_size + buffer_size)
                x_start = max(0, x - buffer_size)
                x_end = min(width, x + window_size + buffer_size)
                
                # Turn the area white
                result[y_start:y_end, x_start:x_end] = [255, 255, 255]
    
    # Save the result
    Image.fromarray(result).save(output_path)
    return output_path

# Usage
# remove_grey_boxes('backend/images/L1_4711_Tobel_page_1.png', window_size=15, buffer_size=10)