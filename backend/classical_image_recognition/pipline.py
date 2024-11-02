import os
import fitz
from pdf2image import convert_from_path

import cv2
import numpy as np

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
    _, bitmap_image = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(output_path, bitmap_image)

    return output_path


import cv2
import numpy as np
import itertools
import math
import os


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

# Example usage
if __name__ == '__main__':
    pdf_path = "backend/testing_pdfs/L1.pdf"
    output_pdf = os.path.join(os.curdir, "backend/temp/Temp.pdf")
    output_png = os.path.join(os.curdir, "backend/temp/Temp.png")

    remove_text_from_pdf(pdf_path, output_pdf)
    image_path = pdf_to_images(output_pdf)

    # Turn PNG image into bit map
    image = read_in_image(image_path)
    image_path = grayscale_to_bitmap(image, output_png)

    # Adjust Grayscale image
    processed_image = preserve_thin_lines(image_path, image_path)

    # Turn image into graph
    # nodes, edges_list, edge_image = image_to_graph(image_path)
    image_path = thicken_everything_2(image_path, output_png)

    image_path = thicken_lines(image_path, output_png)
    # # Turn image into graph
    nodes, edges_list, edge_image = image_to_graph(image_path)

    # Display or save edge_image to visualize detected lines
    cv2.imwrite(output_png, edge_image)
