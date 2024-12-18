import json
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
import json
import matplotlib.pyplot as plt

import json
from collections import defaultdict
from sklearn.cluster import DBSCAN
import numpy as np
from xml.etree import ElementTree as ET


def simplify_svg_graph(svg_file_path, epsilon=200, tolerance=10):
    """
    Simplifies a graph from an SVG file by clustering nodes, removing nearly linear connections,
    and filtering out isolated edges. Saves the output as a JSON file.

    Parameters:
        svg_file_path (str): Path to the SVG file containing the graph.
        epsilon (float): Distance threshold for clustering nodes.
        tolerance (float): Tolerance level for detecting linear alignment in connections.

    Returns:
        str: Path to the saved JSON file.
    """
    # Parse the SVG file and extract line elements
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    svg_edges = []
    for elem in root.findall('{http://www.w3.org/2000/svg}line'):
        # Extract line endpoints as coordinates
        x1, y1 = float(elem.get('x1', 0)), float(elem.get('y1', 0))
        x2, y2 = float(elem.get('x2', 0)), float(elem.get('y2', 0))
        svg_edges.append(((x1, y1), (x2, y2)))

    # Extract unique nodes from edges
    nodes = list({node for edge in svg_edges for node in edge})

    # Cluster nodes using DBSCAN
    node_coords = np.array(nodes)
    dbscan = DBSCAN(eps=epsilon, min_samples=1)
    labels = dbscan.fit_predict(node_coords)

    # Find centroids for each cluster
    cluster_centroids = {}
    for label in set(labels):
        cluster_points = node_coords[labels == label]
        centroid = cluster_points.mean(axis=0)
        cluster_centroids[label] = tuple(centroid)

    # Map edges to cluster centroids
    cluster_edges = set()
    for start, end in svg_edges:
        start_label = labels[nodes.index(start)]
        end_label = labels[nodes.index(end)]
        if start_label != end_label:
            cluster_edges.add((cluster_centroids[start_label], cluster_centroids[end_label]))

    # Helper function to check collinearity
    def are_collinear(p1, p2, p3, tolerance=10):
        area = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0
        return area < tolerance

    # Refine edges to remove nearly linear connections
    refined_edges = set()
    for start, end in cluster_edges:
        keep_connection = True
        for other_point in cluster_centroids.values():
            if other_point != start and other_point != end:
                if are_collinear(start, other_point, end, tolerance=tolerance):
                    keep_connection = False
                    break
        if keep_connection:
            refined_edges.add((start, end))

    # Count connections for each node and remove isolated edges
    connection_count = defaultdict(int)
    for start, end in refined_edges:
        connection_count[start] += 1
        connection_count[end] += 1

    filtered_edges = {
        (start, end) for start, end in refined_edges
        if connection_count[start] > 1 or connection_count[end] > 1
    }

    # Create JSON structure and save it
    simplified_graph = {
        "nodes": [list(node) for node in cluster_centroids.values() if connection_count[node] > 1],
        "edges": [[list(start), list(end)] for start, end in filtered_edges]
    }

    json_file_path = 'filtered_simplified_graph.json'
    with open(json_file_path, 'w') as f:
        json.dump(simplified_graph, f, indent=4)

    return json_file_path


def plot_graph_from_json(json_file_path, save_path):
    """
    Plots a simplified graph based on the JSON file generated by `simplify_svg_graph`.

    Parameters:
        json_file_path (str): Path to the JSON file containing the simplified graph data.
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # Plot the graph
    plt.figure(figsize=(8, 6))

    # Plot edges
    for start, end in edges:
        x_values = [start[0], end[0]]
        y_values = [start[1], end[1]]
        plt.plot(x_values, y_values, 'bo-', markersize=8, linewidth=1)  # 'bo-' shows blue points and connecting lines

    # Label axes and add title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Simplified Graph Plot")

    plt.savefig(save_path)
