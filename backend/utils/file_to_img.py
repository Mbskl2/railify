from pdf2image import convert_from_path
import os

# Function to read PDF and convert each page to an image
def pdf_to_images(pdf_path, output_folder='backend/images', dpi=300):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi)

    # Save each page as an image
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        page.save(image_path, 'PNG')
        print(f"Saved {image_path}")

    return image_path
