from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO("/Users/lukasstrack/BaselHack2024/runs/detect/train2/weights/last.pt")

# Path to your image
image_path = "/Users/lukasstrack/BaselHack2024/dataset/test/images/1cropped_10_L3_17825_Zurichwald_versionB_page_1_png.rf.b426ec363ac93373eafae1402cf9eba3.jpg"

# Perform inference on the image
results = model(image_path)

# Extract results for bounding boxes, class names, and confidence scores
detections = results[0]  # Get the first (and only) result

# Load the image using OpenCV for visualization
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Plot the image and overlay bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

for box in detections.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    conf = box.conf[0]  # Confidence score
    cls = int(box.cls[0])  # Class index
    label = model.names[cls]  # Class label

    # Draw bounding box and label
    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
        )
    )
    plt.text(
        x1,
        y1 - 5,
        f"{label} {conf:.2f}",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="red", alpha=0.5),
    )

plt.axis("off")
plt.show()
