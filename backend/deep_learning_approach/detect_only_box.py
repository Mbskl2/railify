from ultralytics import YOLO
import svgwrite

# Load the YOLOv8 model
model = YOLO("/Users/lukasstrack/BaselHack2024/runs/detect/train2/weights/last.pt")

# Path to your image
image_path = (
    "/Users/lukasstrack/BaselHack2024/croppes_3/2cropped_1_L2_88997_Grosswil_page_3.png"
)

# Perform inference on the image
results = model(image_path)

# Extract detections
detections = results[0]  # Get the first (and only) result

# Load the image to get dimensions
import cv2

image = cv2.imread(image_path)
height, width, _ = image.shape

# Create an SVG drawing
dwg = svgwrite.Drawing("bounding_boxes.svg", size=(width, height))

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
            stroke_width=5,
        )
    )

    # Add label and confidence text
    dwg.add(
        dwg.text(
            f"{label} {conf:.2f}",
            insert=(x1, y1 - 5),
            fill="red",
            font_size=f"100px",
            font_family="Arial",
        )
    )

# Save the SVG file
dwg.save()

print("SVG with bounding boxes and labels created as 'bounding_boxes.svg'.")
