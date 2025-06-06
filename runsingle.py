import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from plyfile import PlyData
from torchvision import transforms

# Import from sam2 for SAM 2.1
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Assuming these are custom modules provided by the user or available in the environment
from classifier import EfficientNetB7Classifier
from util import generate_spherical_image, apply_segmentation_masks

# Define your device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SAM 2.1 Specific Paths and Configuration ---
# IMPORTANT: You need to download the SAM 2.1 checkpoint and its corresponding
# configuration file from the official SAM 2.1 releases or repository.
# Update these paths to your actual file locations.
MODEL_PATH = "Models/checkpoint (1).pt"  # Example: path to your SAM 2.1 checkpoint
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" # Example: path to your SAM 2.1 config file

# Build the SAM 2.1 model
# apply_postprocessing=False is often used here, and post-processing is handled by the mask generator
sam = build_sam2(MODEL_CFG, MODEL_PATH, device=DEVICE, apply_postprocessing=False)
sam.to(device=DEVICE)

# Create the automatic mask generator for SAM 2.1
# SAM 2.1's SAM2AutomaticMaskGenerator takes 'model' as an explicit argument
# and offers additional parameters for fine-tuning mask generation.
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam,
    points_per_side=32,  # Number of points to sample per side of the image
    pred_iou_thresh=0.86,  # Threshold for filtering masks based on predicted IoU
    stability_score_thresh=0.9,  # Threshold for filtering masks based on stability score
    crop_n_layers=1,  # Number of layers to crop the image for multi-scale processing
    crop_n_points_downscale_factor=2, # Factor to downscale image before cropping points
    min_mask_region_area=100,  # Minimum area for a mask region (requires opencv-python)
    use_m2m=True,  # Recommended for SAM2 for improved mask quality
)

# Load the EfficientNetB7 classifier model
model = EfficientNetB7Classifier().to(DEVICE)
model.load_state_dict(torch.load("Models/EffiecientNetB7/best_model (1).pth", map_location=DEVICE))
model.eval()
class_names = model.class_names

# Define the image transformations for the classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define output folders and create them if they don't exist
output_folder_seg = Path("SegmentedImages3")
output_folder_csv = Path("ClassifiedPoints")
output_folder_seg.mkdir(exist_ok=True)
output_folder_csv.mkdir(exist_ok=True)

# Define parameters for bounding box adjustments and image resolution
offset_x, offset_y = 10, 10
min_bbox_size = 20
resolution_y = 500

def process_ply(ply_path):
    """
    Processes a PLY file:
    1. Reads point cloud data and colors.
    2. Rotates the point cloud.
    3. Generates a spherical image projection from the point cloud.
    4. Uses SAM 2.1 to generate segmentation masks on the spherical image.
    5. Classifies segmented objects using a pre-trained EfficientNetB7 model.
    6. Applies segmentation masks to the original image and saves it.
    7. Creates a CSV file with classified 3D points.

    Args:
        ply_path (str): Path to the input PLY file.
    """
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    raw_points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Rotate 90 degrees to the right (clockwise) around Z-axis
    # This rotation matrix rotates around the Y-axis (vertical axis in a typical 3D coordinate system)
    # to align with a common spherical projection orientation.
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    point_cloud = raw_points @ rotation_matrix.T
    colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)

    # Calculate the center of the point cloud for spherical projection
    center_coordinates = point_cloud.mean(axis=0)
    
    # Generate spherical image and mapping from 3D points to 2D pixels
    image, mapping = generate_spherical_image(center_coordinates, colors, point_cloud, resolution_y)
    
    # Create and save a presence map (mask indicating where points are projected)
    presence_map = (mapping != -1).astype(np.uint8) * 255
    cv2.imwrite("mapping_mask.jpg", presence_map)
    cv2.imwrite('original_projection.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Generate segmentation masks using SAM 2.1
    masks = mask_generator.generate(image)
    img_height, img_width = resolution_y, 2 * resolution_y

    class_labels = {}  # Stores object ID to class label mapping
    id_to_pixels = {}  # Stores object ID to pixel coordinates mapping

    # Iterate through each generated mask
    for idx, ann in enumerate(masks):
        ann['id'] = idx  # Assign a unique ID to each annotation
        bbox = ann['bbox']
        x_min, y_min, width, height = map(int, bbox)
        x_max = x_min + width
        y_max = y_min + height

        # Apply offsets to bounding box for better cropping
        x_min = max(0, x_min - offset_x)
        y_min = max(0, y_min - offset_y)
        x_max = min(img_width, x_max + offset_x)
        y_max = min(img_height, y_max + offset_y)

        # Skip invalid or too small bounding boxes
        if x_max <= x_min or y_max <= y_min:
            continue
        if (x_max - x_min) < min_bbox_size or (y_max - y_min) < min_bbox_size:
            continue

        # Crop the object from the spherical image
        cropped_object = image[y_min:y_max, x_min:x_max]
        if cropped_object.size == 0:
            continue
        
        # Convert cropped object to PIL Image for classification
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)).convert("RGB")
        image_tensor = transform(cropped_pil_image).unsqueeze(0).to(DEVICE)

        # Classify the cropped object using the pre-trained EfficientNetB7 model
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            class_label = class_names[predicted_class.item()]

        class_labels[idx] = class_label  # Store the classified label
        
        # Get pixel coordinates for the current mask
        mask = ann['segmentation']
        pixels = np.argwhere(mask)
        # Store pixels associated with this object ID
        id_to_pixels[idx] = pixels if idx not in id_to_pixels else np.vstack((id_to_pixels[idx], pixels))

    # Apply segmentation masks and class labels to the original spherical image for visualization
    segmented_image = apply_segmentation_masks(masks, image, class_labels)
    cv2.imwrite(str(output_folder_seg / "segmented_result.jpg"), cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    classified_points = []
    # Iterate through each object and its pixels to get 3D classified points
    for obj_id, pixels in id_to_pixels.items():
        label = class_labels.get(obj_id, "Unknown") # Get the class label for the object
        for y, x in pixels:
            # Ensure pixel coordinates are within the mapping dimensions
            if 0 <= y < mapping.shape[0] and 0 <= x < mapping.shape[1]:
                point_idx = mapping[y, x] # Get the original 3D point index from the mapping
                if point_idx == -1: # Skip if no 3D point maps to this pixel
                    continue
                
                # Retrieve 3D coordinates and colors
                x_, y_, z_ = point_cloud[point_idx]
                r_, g_, b_ = colors[point_idx]
                
                # Append classified point data
                classified_points.append([x_, y_, z_, r_, g_, b_, label, obj_id])

    # Create a Pandas DataFrame and save to CSV
    df = pd.DataFrame(classified_points, columns=["x", "y", "z", "r", "g", "b", "class", "instance_id"])
    df.to_csv(output_folder_csv / "classified_output.csv", index=False, float_format="%.6f")

if __name__ == "__main__":
    # Example usage: Change to your actual .ply path
    process_ply("stabilized.ply")
    print("\n🎉 PLY file processed successfully.")