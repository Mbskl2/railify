import math

import fitz
import numpy as np

import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from pdf2image import convert_from_path

import cv2
import os

from classical_image_recognition.box_utilites import remove_grey_boxes
from classical_image_recognition.svg_generation import generate_svg_from_lines, save_svg
from classical_image_recognition.line_manipulation import generate_lines, join_horizontal_lines, process_lines
from classical_image_recognition.create_graph import simplify_svg_graph, plot_graph_from_json

import json

# from box_utilites import remove_grey_boxes
# from svg_generation import generate_svg_from_lines, save_svg
# from line_manipulation import generate_lines, process_lines, join_horizontal_lines
# from create_graph import simplify_svg_graph, plot_graph_from_json

from ultralytics import YOLO
import svgwrite

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

def run_preprocessing_pipeline(pdf_path):
    image_path = pdf_to_images(pdf_path)
    return image_path

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


def grayscale_to_bitmap(image, output_path):
    """
    Convert a grayscale image to a binary bitmap based on a threshold, then save it.

    Parameters:
    - image (np.ndarray): The grayscale image to process.
    - output_path (str): The path where the binary image will be saved.
    - threshold (float): Threshold for binarization, between 0 and 1.

    Returns:
    - np.ndarray: The processed binary bitmap image.
    """

    # Apply thresholding
    _, bitmap_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite(output_path, bitmap_image)

    return output_path


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


def crop_image(img_path, output_path, x, y, border_width_x, border_width_y):
    """
    Crops an image based on the given starting point (x, y) and border widths.

    Parameters:
    - img_path (str): Path to the input image.
    - output_path (str): Path where the cropped image will be saved.
    - x (int): The x-coordinate of the top-left corner of the cropping region.
    - y (int): The y-coordinate of the top-left corner of the cropping region.
    - border_width_x (int): The width of the cropping region.
    - border_width_y (int): The height of the cropping region.

    Returns:
    - cropped_image (np.ndarray): The cropped section of the image.
    """
    # Load the image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Define the cropping region
    x_end = x + border_width_x
    y_end = y + border_width_y

    # Ensure the cropping region is within image bounds
    if x < 0 or y < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
        raise ValueError("Cropping region exceeds image boundaries.")

    # Crop the image
    cropped_image = image[y:y_end, x:x_end]

    # Save the cropped image to the specified output path
    cv2.imwrite(output_path, cropped_image)

    return cropped_image


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


def read_svg(file_path):
    try:
        with open(file_path, "r") as file:
            svg_content = file.read()
            soup = BeautifulSoup(svg_content, "xml")  # Parse as XML
        return soup  # Returns the BeautifulSoup object
    except Exception as e:
        print(f"Error reading SVG file: {e}")
        return None


def calculate_distance(points: list):
    points = np.array(points)
    point1 = np.array(points[:2])
    point2 = np.array(points[2:])
    return np.linalg.norm(point2 - point1)


def is_almost_equal(a, b, tol=1e-2):
    """Helper function to check if two floats are approximately equal."""
    return math.isclose(a, b, abs_tol=tol)


def extract_lines_from_svg_with_specific_lenght(svg_string, length_threshold):
    """
    Extracts all line elements from an SVG string as a list of coordinates.

    Parameters:
    - svg_string (str): SVG content as a string.

    Returns:
    - List of lines in the format [[x1, y1, x2, y2]].
    """
    # Parse the SVG string
    root = ET.fromstring(svg_string)

    # Extract namespace if present
    namespace = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    # Find all line elements
    lines = []
    for line in root.findall(f".//{namespace}line"):
        # Extract coordinates as floats
        x1 = float(line.get("x1"))
        y1 = float(line.get("y1"))
        x2 = float(line.get("x2"))
        y2 = float(line.get("y2"))

        distance = calculate_distance([x1, y1, x2, y2])
        vertical = is_almost_equal(x1, x2)

        if distance > length_threshold and not vertical:
            # Append to the lines list
            lines.append((x1, y1, x2, y2))

    return lines


def delete_short_edges(svg_path, output_path, height, width, length_threshold):
    """
    Deletes edges (paths) from an SVG file that are below a specified length threshold.

    Parameters:
    - svg_path (str): Path to the original SVG file.
    - output_path (str): Path where the modified SVG will be saved.
    - length_threshold (float): The minimum length for edges to retain.
    """
    try:
        # Load SVG file
        with open(svg_path, "r") as file:
            svg_content = file.read()
            paths = extract_lines_from_svg_with_specific_lenght(svg_content, length_threshold)
            svg_content = generate_svg_from_lines(paths, width=width, height=height)
            save_svg(svg_content, output_path)

    except Exception as e:
        print(f"Error processing SVG: {e}")


def combine_svgs(svg_path1, svg_path2, output_path):
    # Read the first SVG file
    tree1 = ET.parse(svg_path1)
    root1 = tree1.getroot()

    # Read the second SVG file
    tree2 = ET.parse(svg_path2)
    root2 = tree2.getroot()

    # Adjust namespace for SVG if needed
    ET.register_namespace('', "http://www.w3.org/2000/svg")

    # Append all elements from root2 to root1
    for element in root2:
        root1.append(element)

    # Save the combined SVG to the output path
    tree1.write(output_path)

def run_main_pipeline(pdf_path, border_x, border_y, border_width, border_height):
    output_pdf = os.path.join(os.curdir, "backend/temp/Temp.pdf")
    output_png = os.path.join(os.curdir, "backend/temp/Temp.png")
    output_svg = os.path.join(os.curdir, "backend/temp/Temp.svg")
    cropped_image_path = os.path.join(os.curdir, "backend/temp/Temp_Crop.png")
    model_generated_svg_path = os.path.join(os.curdir, "backend/temp/Temp_Model.svg")
    json_filepath = os.path.join(os.curdir, "backend/temp/Temp.json")
    graph_plot_path = os.path.join(os.curdir, "backend/temp/Temp_graph.png")


    #######################################################################################
    # 1.) HERE: Save Cropped Image for Output
    image_path = pdf_to_images(pdf_path)
    crop_image(image_path, cropped_image_path, border_x, border_y, border_width, border_height)

    #######################################################################################
    # 2.) HERE: Starts Pipeline to get Rail Structure
    remove_text_from_pdf(pdf_path, output_pdf)
    image_path = pdf_to_images(output_pdf)

    crop_image(image_path, image_path, border_x, border_y, border_width, border_height)

    image_path = remove_grey_boxes(image_path, output_png)

    # Turn PNG image into bit map
    image = read_in_image(image_path)
    image_path = grayscale_to_bitmap(image, output_png)

    # Start Removing Artefacts
    preserve_thin_lines(image_path, image_path)

    # Extend Lines to Cover Gaps
    image_path, lines = generate_lines(image_path, output_png)

    # Melt touching Lines Together
    lines = process_lines(lines)

    lines = join_horizontal_lines(lines)

    for i in range(3):
        lines = process_lines(lines, 30, 5)

    # Generate and save SVG
    height, width = image.shape
    svg_text = generate_svg_from_lines(lines, width, height)
    output_svg = save_svg(svg_text, output_svg)

    # Adjust the SVG
    svg_file = read_svg(output_svg)
    # Connect Edges here
    delete_short_edges(output_svg, output_svg, height, width, length_threshold=250)

    ###############################################################################
    # 3.) Load DL Model
    model = YOLO("backend/deep_learning_approach/last_2.pt")

    # Path to your image
    results = model(cropped_image_path)
    # Extract detections
    detections = results[0]  # Get the first (and only) result
    dwg = svgwrite.Drawing(model_generated_svg_path, size=(width, height))

    # Add bounding boxes and labels to the SVG
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class index
        label = model.names[cls]  # Class label

        # Draw bounding box as a rectangle
        dwg.add(
            dwg.rect(
                insert=(x1, y1),
                size=(x2 - x1, y2 - y1),
                fill="none",
                stroke="red",
                stroke_width=10,
            )
        )

        # Add label and confidence text
        dwg.add(
            dwg.text(
                f"{label} {conf:.2f}",
                insert=(x1, y1 - 5),
                fill="red",
                font_size="48px",
                font_family="Arial",
            )
        )

    # Save the SVG file
    dwg.save()

    ###################################################################
    # 4.) Create Graph from SVG
    json_path = simplify_svg_graph(output_svg)
    plot_graph_from_json(json_path, graph_plot_path)

    ###################################################################
    # 5.) Combine the two svgs
    combine_svgs(model_generated_svg_path, output_svg, output_svg)

    return cropped_image_path, output_svg


if __name__ == "__main__":
    path = "backend/files/input_pdfs/L1.pdf"
    run_main_pipeline(path, 500, 1000, 3500, 1500)
