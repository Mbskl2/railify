import numpy as np
from PIL import Image

def is_grey(pixel):
    return pixel < 250

def remove_grey_boxes(image_path, output_path, window_size=15, buffer_size=10, scale_factor=0.5):
    # Open the image
    img = Image.open(image_path)
    
    # Get original size
    original_size = img.size
    
    # Scale down
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    img_small = img.resize(new_size, Image.LANCZOS)
    
    img_array = np.array(img_small)

    greyscale = np.mean(img_array, axis=2).astype(np.uint8)
    
    height, width, _ = img_array.shape
    
    # Create a copy of the image to modify
    result = np.copy(img_array)
    
    # Adjust window_size and buffer_size for the scaled image
    scaled_window_size = max(1, int(window_size * scale_factor))
    scaled_buffer_size = max(1, int(buffer_size * scale_factor))
    
    for y in range(0, height - scaled_window_size + 1, scaled_window_size):
        for x in range(0, width - scaled_window_size + 1, scaled_window_size):
            window = greyscale[y:y+scaled_window_size, x:x+scaled_window_size]
            
            # Check if all pixels in the window are grey
            if np.all([is_grey(pixel) for row in window for pixel in row]):
                # Calculate the area to turn white, including the buffer
                y_start = max(0, y - scaled_buffer_size)
                y_end = min(height, y + scaled_window_size + scaled_buffer_size)
                x_start = max(0, x - scaled_buffer_size)
                x_end = min(width, x + scaled_window_size + scaled_buffer_size)
                
                # Turn the area white
                result[y_start:y_end, x_start:x_end] = [255, 255, 255]
    
    # Convert back to PIL Image
    result_img = Image.fromarray(result)
    
    # Scale back up to original size
    result_img = result_img.resize(original_size, Image.LANCZOS)
    
    # Save the result
    result_img.save(output_path)
    return output_path

#remove_grey_boxes('backend/images/L1_4711_Tobel_page_1.png', window_size=15, buffer_size=10)