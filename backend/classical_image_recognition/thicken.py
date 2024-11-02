import cv2
import numpy as np

def thicken_lines(image_path='backend/images/L1_66666_Schoenwald_page_1.png', output_path='result.png'):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image file or path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 500, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)

    # Filter and draw thickened lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=15)

    cv2.imwrite(output_path, image)
    return output_path
    # cv2.imshow('Detected Lines', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def thicken_everything(file_path='backend/images/L1_66666_Schoenwald_page_1.png', output_path='result.png'):
    import cv2
    import numpy as np

    # Read the PNG image
    image = cv2.imread(file_path)

    if image is None:
        print("Error: Image not found or cannot be opened.")
        return

    # Check if the image has an alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Image has an alpha channel
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Image doesn't have an alpha channel (RGB)
        bgr = image
        alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
    else:
        print("Error: Unexpected image format.")
        return

    # Split the image into color and alpha channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]

    # Convert the color channels to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Create a binary mask of the black areas
    _, black_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Create a kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    # Dilate the black areas
    dilated_mask = cv2.dilate(black_mask, kernel, iterations=1)

    # Invert the dilated mask
    inverted_mask = cv2.bitwise_not(dilated_mask)

    # Apply the inverted mask to the original color channels
    result_bgr = cv2.bitwise_and(bgr, bgr, mask=inverted_mask)

    # Combine the result with the alpha channel
    result = cv2.merge([result_bgr, alpha])

    # Save the result
    cv2.imwrite(output_path, result)
    return output_path
    # cv2.imshow('Detected Lines', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def thicken_everything_2(file_path='backend/images/L2_22345_Beinwil_page_1.png', output_path='result.png'):
    from PIL import Image, ImageDraw

    def expand_dark_areas(input_path, output_path, expansion=5, darkness_threshold=30):
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGBA if it's not already
            img = img.convert("RGBA")
            
            # Create a new image with a transparent background
            new_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            
            # Create a draw object
            draw = ImageDraw.Draw(new_img)
            
            # Iterate through each pixel
            for x in range(img.width):
                for y in range(img.height):
                    r, g, b, a = img.getpixel((x, y))
                    
                    # If the pixel is dark (close to black)
                    if max(r, g, b) < darkness_threshold and a > 0:
                        # Draw a filled black circle at this position
                        draw.ellipse([x-expansion, y-expansion, 
                                    x+expansion, y+expansion], 
                                    fill=(0, 0, 0, a))
            
            # Composite the new image over the original
            result = Image.alpha_composite(img, new_img)
            
            # Save the result
            result.save(output_path)

# Usage
    expand_dark_areas(file_path, output_path, expansion=7, darkness_threshold=30)
    return output_path

#thicken_everything_2()
#thicken_lines()

#===
# line_mask = np.zeros(image.shape[:2], dtype=np.uint8)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=20)

# # Dilate the lines to make them thicker
# kernel = np.ones((5,5), np.uint8)
# thickened_lines = cv2.dilate(line_mask, kernel, iterations=1)

# # Create a colored mask for overlay
# colored_mask = cv2.cvtColor(thickened_lines, cv2.COLOR_GRAY2BGR)
# colored_mask[np.where((colored_mask == [255,255,255]).all(axis=2))] = [0,0,255]  # Red color

# # Overlay the thickened lines on the original image
# result = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

# cv2.imshow('Original', image)
# cv2.imshow('Thickened Lines', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#---
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply edge detection
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# # Detect lines using Hough Transform
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# # Filter and draw thickened lines
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         # Calculate line length
#         length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
#         # Filter lines based on length (e.g., longer than 150 pixels)
#         if length > 100:
#             # Draw thickened line
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

# # Save the result
# cv2.imwrite('output_image.jpg', image)

# # Display the result (optional)
# cv2.imshow('Detected Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply edge detection
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imshow('Detected Lines', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np

# def findStraightLines(img, rho, theta, threshold, min_line_length, max_line_gap):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur_gray, 50, 150)
    
#     lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                             min_line_length, max_line_gap)
    
#     line_image = np.copy(img) * 0
    
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
#     lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
#     return lines_edges, lines


# # Set parameters for line detection
# rho = 1  # Distance resolution in pixels
# theta = np.pi / 180  # Angular resolution in radians (1 degree)
# threshold = 20  # Minimum number of intersections to detect a line
# min_line_length = 50  # Minimum number of pixels making up a line
# max_line_gap = 10  # Maximum gap in pixels between connectable line segments

# # Detect lines
# lines_edges, lines = findStraightLines(image, rho, theta, threshold, min_line_length, max_line_gap)

# # # Display the result
# cv2.imshow('Detected Lines', lines_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()