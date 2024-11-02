import os
import cv2
import numpy as np
import fitz
from pdf2image import convert_from_path


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


def grayscale_to_bitmap(image, output_path, threshold=0.5):
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

