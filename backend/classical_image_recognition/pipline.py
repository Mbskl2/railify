import os
import fitz
from pdf2image import convert_from_path

from thicken import thicken_everything, thicken_everything_2, thicken_lines

import cv2
import numpy as np
import itertools
import math
import os

from svg_generation import generate_svg_from_lines, save_svg
from line_manipulation import extend_lines
from thicken import thicken_everything, thicken_everything_2, thicken_lines


# Setup pipline to read in Image and Extract Lines
def pdf_to_images(pdf_path, output_folder='backend/temp', dpi=300, name="page"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)

    # Save each page as an image
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f'{name}_{i + 1}.png')
        page.save(image_path, 'PNG')
        print(f"Saved {image_path}")

    return image_path

def read_in_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def remove_text_from_pdf(input_pdf_path, output_pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(input_pdf_path)

    # Iterate over each page and redact text
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        # Get all words on the page; each word is represented by its bounding box
        words = page.get_text("words")  # Returns list of tuples (x0, y0, x1, y1, "word")

        # Redact each word area
        for word in words:
            rect = fitz.Rect(word[:4])  # Get the bounding box for the word
            page.add_redact_annot(rect)  # Add a redaction annotation to the area
            page.apply_redactions()  # Apply the redaction (turns area blank)

    # Save the modified PDF
    pdf_document.save(output_pdf_path)
    pdf_document.close()
    print(f"Text removed. Saved new PDF as '{output_pdf_path}'")


def grayscale_to_bitmap(image, output_path, threshold=0.8):
    """
    Convert a grayscale image to a binary bitmap based on a threshold, then save it.

    Parameters:
    - image (np.ndarray): The grayscale image to process.
    - output_path (str): The path where the binary image will be saved.
    - threshold (float): Threshold for binarization, between 0 and 1.

    Returns:
    - np.ndarray: The processed binary bitmap image.
    """
    # Ensure the threshold is within the 0-255 range
    thresh_value = int(threshold * 255)

    # Apply thresholding
    _, bitmap_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(output_path, bitmap_image)

    return output_path

def is_right_angle(p1, p2, p3):
    """
    Check if the angle formed by three points is approximately 90 degrees.
    """

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    a = distance(p1, p2)
    b = distance(p2, p3)
    c = distance(p1, p3)

    # Check if the points form a right triangle (Pythagorean theorem)
    return abs(a ** 2 + b ** 2 - c ** 2) < 1e-2


def filter_rectangles(nodes, edges_list):
    """
    Filter out rectangles based on nodes and edges.
    """
    # Convert edges list to a set for easy access
    edges_set = set(edges_list)
    rectangles = []

    # Iterate over combinations of four nodes to find rectangles
    for quad in itertools.combinations(nodes, 4):
        # Check if they form a rectangle by looking at each pair of points
        pairs = list(itertools.combinations(quad, 2))
        lines = [pair for pair in pairs if pair in edges_set or (pair[1], pair[0]) in edges_set]

        if len(lines) == 4:
            # Check if the four points form right angles
            p1, p2, p3, p4 = quad
            if (is_right_angle(p1, p2, p3) and is_right_angle(p2, p3, p4) and
                is_right_angle(p3, p4, p1) and is_right_angle(p4, p1, p2)):
                rectangles.append(lines)

    # Remove rectangle edges from edges_list
    for rect in rectangles:
        for edge in rect:
            if edge in edges_list:
                edges_list.remove(edge)
            elif (edge[1], edge[0]) in edges_list:
                edges_list.remove((edge[1], edge[0]))

    # Filter out nodes that are only part of rectangles
    new_nodes = [node for node in nodes if any(edge for edge in edges_list if node in edge)]

    return new_nodes, edges_list


def image_to_graph(img_path, edge_threshold1=200, edge_threshold2=300, min_line_length=100, max_line_gap=10):
    """
    Converts a grayscale image into a graph representation of nodes and edges using edge and line detection,
    removes any graph components that consist of only two nodes and one edge, and filters out rectangles.

    Parameters:
    - img_path (str): Path to the grayscale image.
    - edge_threshold1 (int): First threshold for the Canny edge detector.
    - edge_threshold2 (int): Second threshold for the Canny edge detector.
    - min_line_length (int): Minimum length of a line for Hough line detection.
    - max_line_gap (int): Maximum gap between points on the same line for Hough line detection.

    Returns:
    - nodes (list): List of nodes (approximate coordinates of intersections).
    - edges (list): List of edges (line segments connecting nodes).
    - edge_image (np.ndarray): Image showing detected edges for visualization.
    """
    # Verify that the provided path is valid
    if not isinstance(img_path, str) or not os.path.isfile(img_path):
        raise ValueError("Invalid image path. Please provide a valid string path to an existing image file.")

    # Load the grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Apply Canny edge detection with the specified thresholds
    edges = cv2.Canny(image, edge_threshold1, edge_threshold2)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Dictionary to hold nodes and edges in an adjacency list format
    adjacency_list = {}
    edge_image = np.zeros_like(image)
    nodes = []
    edges_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            start_node = (x1, y1)
            end_node = (x2, y2)

            # Add nodes and edges to adjacency list
            if start_node not in adjacency_list:
                adjacency_list[start_node] = set()
            if end_node not in adjacency_list:
                adjacency_list[end_node] = set()

            adjacency_list[start_node].add(end_node)
            adjacency_list[end_node].add(start_node)

            # Draw the detected edge on edge_image for visualization
            cv2.line(edge_image, start_node, end_node, 255, 1)

    # Filter out nodes and edges that are part of a two-node, single-edge graph
    for node, connections in adjacency_list.items():
        if len(connections) == 1:
            connected_node = list(connections)[0]
            if len(adjacency_list[connected_node]) == 1:
                # Both nodes are only connected to each other, so ignore this edge
                continue

        # Add the valid node and its edges
        if node not in nodes:
            nodes.append(node)
        for connected_node in connections:
            edge = (node, connected_node)
            reverse_edge = (connected_node, node)
            # Only add each edge once
            if edge not in edges_list and reverse_edge not in edges_list:
                edges_list.append(edge)

    # Remove rectangles
    nodes, edges_list = filter_rectangles(nodes, edges_list)

    return nodes, edges_list, edge_image

def preserve_thin_lines(img_path, output_path, blob_size=10):
    """
    Processes a grayscale bitmap image to remove thicker blobs or symbols while preserving thin lines.

    Parameters:
    - img_path (str): Path to the grayscale bitmap image.
    - output_path (str): Path where the processed image will be saved.
    - blob_size (int): Size of the kernel for removing thicker blobs (larger removes larger blobs).

    Returns:
    - processed_image (np.ndarray): The processed image with blobs removed and thin lines preserved.
    """
    # Load the grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Ensure the image is binary (0 and 255 only)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Invert the image so that black parts (foreground) become white for easier blob removal
    inverted_image = cv2.bitwise_not(binary_image)

    # Remove blobs using morphological opening with a larger circular kernel
    blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blob_size, blob_size))
    blobs_removed = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, blob_kernel)

    # Preserve thin lines by subtracting the blobs from the original inverted image
    thin_lines_only = cv2.subtract(inverted_image, blobs_removed)

    # Invert the result back to black on white
    processed_image = cv2.bitwise_not(thin_lines_only)

    # Save the processed image to the specified output path
    cv2.imwrite(output_path, processed_image)

    return processed_image

def run_preprocessing_pipeline(pdf_path):
    image_path = pdf_to_images(pdf_path)
    return image_path

def run_main_pipeline(image_path, border_x, border_y, border_width, border_height):

def fill_line_interruptions(img_path, output_path, gap_size=50):
    """
    Processes a grayscale image to fill interruptions in horizontal and vertical lines only.

    Parameters:
    - img_path (str): Path to the grayscale bitmap image.
    - output_path (str): Path where the processed image will be saved.
    - gap_size (int): Maximum gap size to fill (larger values bridge larger gaps).

    Returns:
    - processed_image (np.ndarray): The processed image with interruptions in horizontal and vertical lines filled.
    """
    # Load the grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Invert the image so that black lines become white for easier morphological operations
    inverted_image = cv2.bitwise_not(binary_image)

    # Fill horizontal gaps
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_size, 1))
    horizontal_filled = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, horizontal_kernel)

    # Fill vertical gaps
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_size))
    vertical_filled = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, vertical_kernel)

    # Combine both horizontal and vertical results
    combined_filled = cv2.bitwise_or(horizontal_filled, vertical_filled)

    # Invert back to original black-on-white format
    processed_image = cv2.bitwise_not(combined_filled)

    # Save the processed image to the specified output path
    cv2.imwrite(output_path, processed_image)

    return processed_image
def resize_image(img_path, output_path, width=1920, height=1080):
    """
    Resizes the image to the specified width and height.

    Parameters:
    - img_path (str): Path to the input image.
    - output_path (str): Path where the resized image will be saved.
    - width (int): The target width of the resized image.
    - height (int): The target height of the resized image.

    Returns:
    - resized_image (np.ndarray): The resized image.
    """
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Resize the image to the specified dimensions
    resized_image = cv2.resize(image, (width, height))

    # Save the resized image to the specified output path
    cv2.imwrite(output_path, resized_image)

    return resized_image

def remove_rectangles(img_path, output_path, min_aspect_ratio=0.8, max_aspect_ratio=1.2):
    """
    Processes a binary image to detect and remove rectangular shapes.

    Parameters:
    - img_path (str): Path to the binary (black and white) image.
    - output_path (str): Path where the processed image will be saved.
    - min_aspect_ratio (float): Minimum aspect ratio to consider a contour as a rectangle.
    - max_aspect_ratio (float): Maximum aspect ratio to consider a contour as a rectangle.

    Returns:
    - processed_image (np.ndarray): The processed image with rectangles removed.
    """
    # Load the grayscale image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the binary image to draw over rectangles
    processed_image = binary_image.copy()

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 vertices (indicating a possible rectangle)
        if len(approx) == 4:
            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Check if the aspect ratio is within the range for rectangles
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # Fill the rectangle area with white (removing it from the image)
                cv2.drawContours(processed_image, [contour], -1, (255), thickness=cv2.FILLED)

    # Save the processed image to the specified output path
    cv2.imwrite(output_path, processed_image)

    return processed_image


def remove_small_artifacts(image_path, output_path, artifact_size_threshold=10):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding to make the artifacts more distinguishable
    _, binary_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the artifacts
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to remove small artifacts
    mask = np.ones_like(binary_img) * 255  # Start with a white mask

    for contour in contours:
        # Only keep contours larger than the threshold
        if cv2.contourArea(contour) > artifact_size_threshold:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)

    # Apply the mask to the original image to remove small artifacts
    cleaned_img = cv2.bitwise_and(img, img, mask=mask)

    # Save the result
    cv2.imwrite(output_path, cleaned_img)
    print(f"Cleaned image saved to {output_path}")


def remove_large_black_blocks(image_path, output_path, block_size_threshold=10):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding to highlight black areas
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the black blocks
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to remove large black blocks
    mask = np.ones_like(binary_img) * 255  # Start with a white mask

    for contour in contours:
        # Remove contours larger than the threshold (considered large black blocks)
        if cv2.contourArea(contour) > block_size_threshold:
            cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)

    # Apply the mask to the original image to remove large black blocks
    cleaned_img = cv2.bitwise_and(img, img, mask=mask)

    # Save the result
    cv2.imwrite(output_path, cleaned_img)
    print(f"Cleaned image with large black blocks removed saved to {output_path}")


def thicken_black_pixels(image_path, output_path, kernel_size=5, iterations=10):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding to ensure black areas are clearly defined
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to thicken black pixels
    thickened_img = cv2.dilate(binary_img, kernel, iterations=iterations)

    # Invert the image back to match original colors
    thickened_img = cv2.bitwise_not(thickened_img)

    # Save the result
    cv2.imwrite(output_path, thickened_img)
    print(f"Image with thickened black pixels saved to {output_path}")


def run_preprocessing_pipeline(pdf_path):
    image_path = pdf_to_images(pdf_path)
    return image_path

def run_main_pipeline(image_path, border_x, border_y, border_width, border_height):
    output_pdf = os.path.join(os.curdir, "backend/temp/Temp.pdf")
    output_png = os.path.join(os.curdir, "backend/temp/Temp.png")
    output_svg = os.path.join(os.curdir, "backend/temp/Temp.svg")

    remove_text_from_pdf(pdf_path, output_pdf)
    image_path = pdf_to_images(output_pdf)

    # Turn PNG image into bit map
    image = read_in_image(image_path)
    image_path = grayscale_to_bitmap(image, output_png)

    # Remove Rectangles
    remove_rectangles(image_path, image_path)

    # Adjust Grayscale image
    preserve_thin_lines(image_path, image_path)

    # Thicken
    thicken_black_pixels(image_path, image_path)

    # Remove Artifacts and blck bocks
    # remove_small_artifacts(image_path, image_path)
    # remove_large_black_blocks(image_path, image_path)

    # Resize image
    # resize_image(image_path, output_png)

    # Fill in Gaps
    # fill_line_interruptions(image_path, output_png)

    # Turn image into graph
    # nodes, edges_list, edge_image = image_to_graph(image_path)
    image_path = thicken_everything_2(image_path, output_png)

    image_path, lines = extend_lines(image_path, output_png)

    height, width = image.shape
    svg_text = generate_svg_from_lines(lines, width, height)
    save_svg(svg_text)
    #image_path = thicken_lines(image_path, output_png)

    # # Turn image into graph
    #nodes, edges_list, edge_image = image_to_graph(image_path)

    # Display or save edge_image to visualize detected lines
    #cv2.imwrite(output_png, edge_image)
