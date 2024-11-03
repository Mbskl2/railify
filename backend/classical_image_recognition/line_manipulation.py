import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from shapely import minimum_rotated_rectangle

def group_lines(lines, angle_threshold=1, distance_threshold=10):
    groups = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        found_group = False
        for group in groups:
            group_angle = group['angle']
            if abs(angle - group_angle) < angle_threshold:
                # Check if the line is close to any line in the group
                for group_line in group['lines']:
                    gx1, gy1, gx2, gy2 = group_line[0]
                    if (euclidean((x1, y1), (gx1, gy1)) < distance_threshold or
                        euclidean((x2, y2), (gx2, gy2)) < distance_threshold):
                        group['lines'].append(line)
                        found_group = True
                        break
            if found_group:
                break
        
        if not found_group:
            groups.append({'angle': angle, 'lines': [line]})
    
    return groups

import math

def extend_line(line, extension):
    x1, y1, x2, y2 = line
    
    # Calculate line vector
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate line length
    length = math.sqrt(dx**2 + dy**2)
    
    # Normalize vector
    if length > 0:
        dx /= length
        dy /= length
    
    # Scale vector by extension amount
    dx *= extension
    dy *= extension
    
    # Extend both endpoints
    new_x1 = x1 - dx
    new_y1 = y1 - dy
    new_x2 = x2 + dx
    new_y2 = y2 + dy
    
    return (new_x1, new_y1, new_x2, new_y2)

def bulk_out_lines(lines):
    extended_lines = [[extend_line(tuple(line[0]), 1)] for line in lines]
    return extended_lines

def generate_lines(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
    
    def calculate_angle(line):
        x1, y1, x2, y2 = line[0]
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Convert to degrees

    def are_lines_close(line1, line2, angle_tolerance=5, distance_tolerance=30):
        angle1 = calculate_angle(line1)
        angle2 = calculate_angle(line2)

        # Check angle similarity
        if abs(angle1 - angle2) > angle_tolerance:
            return False

        # Check distance between endpoints
        x1_1, y1_1, x2_1, y2_1 = line1[0]
        x1_2, y1_2, x2_2, y2_2 = line2[0]

        distances = [
            np.sqrt((x1_1 - x1_2)**2 + (y1_1 - y1_2)**2),
            np.sqrt((x1_1 - x2_2)**2 + (y1_1 - y2_2)**2),
            np.sqrt((x2_1 - x1_2)**2 + (y2_1 - y1_2)**2),
            np.sqrt((x2_1 - x2_2)**2 + (y2_1 - y2_2)**2),
        ]
        
        return any(d < distance_tolerance for d in distances)

    # Function to merge close parallel lines
    def merge_lines(lines):
        merged_lines = []
        
        for line in lines:
            found_match = False
            
            for merged_line in merged_lines:
                if are_lines_close(line, merged_line):
                    # Merge by extending the existing line
                    x1_new = min(line[0][0], merged_line[0][0])
                    y1_new = min(line[0][1], merged_line[0][1])
                    x2_new = max(line[0][2], merged_line[0][2])
                    y2_new = max(line[0][3], merged_line[0][3])
                    
                    merged_line[0] = [x1_new, y1_new, x2_new, y2_new]
                    found_match = True
                    break
            
            if not found_match:
                merged_lines.append(line)
        
        return np.array(merged_lines)

    
    print(lines)
    for i in range(30):
        lines = bulk_out_lines(lines)
    merged_parallel_lines = lines
    print(merged_parallel_lines)

     # Group lines
    groups = group_lines(merged_parallel_lines)
    
    # Merge line segments within each group
    return_lines = []
    for group in groups:
        merged_lines = merge_line_segments(group)
        
        # Draw merged lines
        for line in merged_lines:
            return_lines.append(line)
            x1, y1, x2, y2 = line
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
    
    # # Display the result
    # cv2.imshow('Merged Lines', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(return_lines)
    cv2.imwrite(output_path, img)
    return output_path, return_lines


def merge_line_segments(group):
    merged_lines = []
    for line in group['lines']:
        x1, y1, x2, y2 = line[0]
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        merged_lines.append((x1, y1, x2, y2))
    
    merged_lines.sort()
    result = []
    for line in merged_lines:
        if not result or line[0] > result[-1][2]:
            result.append(line)
        else:
            result[-1] = (result[-1][0], result[-1][1], max(result[-1][2], line[2]), line[3])
    
    return result

# # Usage
# image_path = 'backend/temp/temp.png'
# extend_lines(image_path, 'result.png')

#def join_large_lines(lines):
    
import math
import numpy as np
from shapely.geometry import box, Polygon

def create_bounding_box(line, width=30):
    x1, y1, x2, y2 = line
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = math.atan2(y2 - y1, x2 - x1)
    
    dx = width / 2 * math.sin(angle)
    dy = width / 2 * math.cos(angle)
    
    corners = [
        (x1 - dx, y1 + dy),
        (x1 + dx, y1 - dy),
        (x2 + dx, y2 - dy),
        (x2 - dx, y2 + dy)
    ]
    return Polygon(corners)

def group_lines_by_angle(lines, angle_threshold=2):
    grouped_lines = {}
    for line in lines:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        
        found_group = False
        for group_angle in grouped_lines:
            if abs(angle - group_angle) <= angle_threshold:
                grouped_lines[group_angle].append(line)
                found_group = True
                break
        
        if not found_group:
            grouped_lines[angle] = [line]
    
    return grouped_lines

from shapely.geometry import box
from collections import defaultdict

def group_overlapping_boxes(boxes):
    # Create a dictionary to store the groups
    groups = defaultdict(set)
    
    # Assign each box to a group
    for i, box1 in enumerate(boxes):
        groups[i].add(i)
        for j, box2 in enumerate(boxes[i+1:], start=i+1):
            if box1.intersects(box2):
                groups[i].add(j)
                groups[j].add(i)
    
    # Merge overlapping groups
    merged_groups = []
    processed = set()
    
    for i, group in groups.items():
        if i not in processed:
            merged_group = set(group)
            to_process = list(group)
            
            while to_process:
                j = to_process.pop()
                if j not in processed:
                    merged_group.update(groups[j])
                    to_process.extend(groups[j])
                    processed.add(j)
            
            merged_groups.append([boxes[i] for i in merged_group])
    
    return merged_groups


from shapely.geometry import box, LineString
from shapely.ops import unary_union
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely import minimum_rotated_rectangle

def get_middle_line(boxes):
    # Create a bounding box around all the given boxes
    union_of_boxes = unary_union(boxes)
    rotated_bounding_box = minimum_rotated_rectangle(union_of_boxes)

        # Get the coordinates of the rotated box
    coords = list(rotated_bounding_box.exterior.coords)
    
    # Calculate the lengths of all sides
    sides = [LineString([coords[i], coords[i+1]]) for i in range(4)]
    lengths = [side.length for side in sides]
    
    # Find the indices of the two shorter sides
    short_side_indices = sorted(range(4), key=lambda i: lengths[i])[:2]
    
    # Calculate midpoints of the shorter sides
    midpoints = [Point(sides[i].interpolate(0.5, normalized=True)) for i in short_side_indices]
    
    # Create a line between these midpoints
    middle_line = LineString(midpoints)
    
    # Extract start and end coordinates
    start_x, start_y = middle_line.coords[0]
    end_x, end_y = middle_line.coords[1]
    
    return (start_x, start_y, end_x, end_y)

def process_lines(lines, width=30, angle_threshold=2):
    grouped_lines = group_lines_by_angle(lines, angle_threshold)
    results = {}
    
    for angle, group in grouped_lines.items():
        bounding_boxes = [create_bounding_box(line, width) for line in group]
        merged_boxes = group_overlapping_boxes(bounding_boxes)
        results[angle] = merged_boxes

    new_lines = []
    for group in results.values():
        for boxes in group:
            new_line = get_middle_line(boxes)
            new_lines.append(new_line)

    return new_lines