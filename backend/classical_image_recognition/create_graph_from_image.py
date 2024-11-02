import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load the image
img_path = "backend/images/L1_4711_Tobel_page_1.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded correctly
if img is None:
    raise FileNotFoundError(f"Image at path {img_path} could not be loaded. Please check the path and try again.")

# Step 1: Adaptive Thresholding to Isolate Main Structures
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Step 2: Morphological Operations to Clean Up Noise
kernel = np.ones((4, 4), np.uint8)
processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Step 3: Detect Contours and Filter by Size
contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Create a blank image to draw contours on
# Make sure it has the same dimensions as your original image
contour_image = np.zeros(processed.shape, dtype=np.uint8)

# Draw all contours on the blank image
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)


# Display the image
cv2.imshow("Contours", contour_image)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()

# # Create a NetworkX graph
# G = nx.Graph()

# # Define a minimum area threshold to filter out small contours
# min_contour_area = 100  # Adjust this value based on your map scale

# # Loop through each contour to find nodes
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > min_contour_area:
#         # Calculate the center of each contour
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             # Add each center point as a node
#             G.add_node((cx, cy))

# # Step 4: Connect Nearby Nodes
# # Define a maximum distance threshold for connecting nodes
# max_distance = 150  # Adjust based on map scale
# nodes = list(G.nodes)

# # Connect each node to nearby nodes within max_distance
# for i, node1 in enumerate(nodes):
#     for node2 in nodes[i+1:]:
#         distance = np.linalg.norm(np.array(node1) - np.array(node2))
#         if distance < max_distance:
#             G.add_edge(node1, node2)

# # Step 5: Ensure a Single Connected Graph
# # Find the largest connected component and keep only that
# largest_cc = max(nx.connected_components(G), key=len)
# G = G.subgraph(largest_cc).copy()

# # Step 6: Draw the Resulting Graph
# output_image = np.zeros_like(img)

# # Draw edges in the connected graph
# for edge in G.edges:
#     cv2.line(output_image, edge[0], edge[1], 255, 1)

# # Draw nodes
# for node in G.nodes:
#     cv2.circle(output_image, node, 5, 255, -1)

# # Display the final connected graph
# plt.figure(figsize=(10, 10))
# plt.imshow(output_image, cmap="gray")
# plt.title("Single Connected Graph Representation of Train Map")
# plt.axis("off")
# plt.show()
