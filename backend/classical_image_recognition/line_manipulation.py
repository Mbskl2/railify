import cv2
import numpy as np
from scipy.spatial.distance import euclidean

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

def extend_lines(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
    print(lines)

    # Function to calculate distance between two points
    def distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Function to merge close parallel lines
    def merge_lines(lines, max_distance=30):
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            found_match = False
            for merged_line in merged_lines:
                mx1, my1, mx2, my2 = merged_line[0]
                m_angle = np.arctan2(my2 - my1, mx2 - mx1)
                
                # Check if lines are parallel (similar angle)
                if abs(angle - m_angle) < np.pi / 36:  # 5 degrees tolerance
                    # Check if lines are close
                    if (distance(x1, y1, mx1, my1) < max_distance or
                        distance(x2, y2, mx2, my2) < max_distance):
                        # Merge lines by extending the existing line
                        merged_line[0] = [min(x1, mx1), min(y1, my1),
                                        max(x2, mx2), max(y2, my2)]
                        found_match = True
                        break
            
            if not found_match:
                merged_lines.append(line)

        return np.array(merged_lines)

    merged_parallel_lines = merge_lines(lines)
    print(merged_parallel_lines)

     # Group lines
    groups = group_lines(merged_parallel_lines)
    
    # Merge line segments within each group
    for group in groups:
        merged_lines = merge_line_segments(group)
        
        # Draw merged lines
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
    
    # Display the result
    cv2.imshow('Merged Lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output_path


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

# Usage
image_path = 'backend/temp/temp.png'
extend_lines(image_path, 'result.png')