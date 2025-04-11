import numpy as np
import cv2
class_colors = {
    "Beam": [0, 128, 128],  # Teal
    "Ceiling": [0, 255, 255],  # Cyan
    "Column": [0, 0, 255],  # Blue
    "Ductwork": [128, 0, 0],  # Dark Red
    "Floor": [0, 255, 0],  # Green
    "Pipe": [255, 0, 255],  # Magenta
    "Stairs": [255, 255, 0],  # Yellow
    "Wall": [255, 0, 0]  # Red
}

def generate_spherical_image(center_coordinates, colors, region_points,resolution_y=500):
    # Translate the point cloud by the negation of the center coordinates
    translated_points = region_points - center_coordinates

    # Convert 3D point cloud to spherical coordinates
    theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
    norms = np.linalg.norm(translated_points, axis=1)

    norms[norms == 0] = 1e-6
    phi = np.arccos(np.clip(translated_points[:, 2] / norms, -1.0, 1.0))

    # Map spherical coordinates to pixel coordinates
    x = (theta + np.pi) / (2 * np.pi) * (2 * resolution_y)
    y = phi / np.pi * resolution_y

     # Create the spherical image with RGB channels
    resolution_x = 2 * resolution_y
    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

    # Create the mapping between point cloud and image coordinates
    mapping = np.full((resolution_y, resolution_x), -1, dtype=int)
    num_invalid_points = np.sum(mapping == -1)
    print(f"Number of invalid points: {num_invalid_points}")
    # Assign points to the image pixels
    for i in range(len(translated_points)):
        ix = np.clip(int(x[i]), 0, resolution_x - 1)
        iy = np.clip(int(y[i]), 0, resolution_y - 1)
        if mapping[iy, ix] == -1 or np.linalg.norm(translated_points[i]) < np.linalg.norm(translated_points[mapping[iy, ix]]):
            mapping[iy, ix] = i
            image[iy, ix] = colors[i]
    return image,mapping
def compute_bounding_box_center(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return (min_coords + max_coords) / 2

def apply_segmentation_masks(masks, image, class_labels):
    overlay = image.copy()
    
    for ann in masks:
        bbox = ann['bbox']
        obj_id = ann['id']
        
        # Safely retrieve class label, defaulting to "Unknown" if not found
        obj_class = class_labels.get(obj_id, "Unknown")
        color = class_colors.get(obj_class, [255, 255, 255])  

        # Draw mask overlay
        mask = ann['segmentation'].astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
    
    return overlay