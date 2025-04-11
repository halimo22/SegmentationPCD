import os
import re
import cv2
import numpy as np
import pandas as pd
import laspy
from PIL import Image
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from torchvision.models import efficientnet_b7
import torch
from torch import nn
import torchvision.transforms as transforms
from util import generate_spherical_image, apply_segmentation_masks
from classifier import EfficientNetB7Classifier


# ========== Model ==========
MODEL = "Models/sam_vit_h_4b8939.pth"
USED_D = "cuda:0"
sam = sam_model_registry["vit_h"](checkpoint = MODEL)
sam.to(device = USED_D)

mask_generator = SamAutomaticMaskGenerator(
    sam, 
    pred_iou_thresh=0.75,
    stability_score_thresh=0.80
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = EfficientNetB7Classifier().to(device)  
model.load_state_dict(torch.load(r"Models\EffiecientNetB7\best_model.pth", map_location=device))
model.eval()


class_names=model.class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Configuration ==========
csv_file = r"PrepedScans\BestBuilding\scan_positions.csv"
folder1 = r"PrepedScans\BestBuilding"       # LAS files
folder2 = r"SegmentedImages2"               # Output segmented images
folder3 = r"PointCloudcolored"              # Output LAS
original_images_folder = Path("PrepedScans\BestBuilding\SphericalImages")

offset_x = 10
offset_y = 10
min_bbox_size = 20
resolution_y = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scan_positions_df = pd.read_csv(csv_file)
scan_positions_df['Scan_ID'] = scan_positions_df['Scan_ID'].astype(int)

las_files = {int(os.path.splitext(f)[0]): f for f in os.listdir(folder1) if f.endswith(".las")}
original_image_files = {int(re.search(r'filtered_scan_(\d+)', f).group(1)): f for f in os.listdir(original_images_folder) if re.search(r'filtered_scan_(\d+)', f)}
common_keys = sorted(set(las_files.keys()).intersection(original_image_files.keys()))

for scan_number in common_keys:
    print(f"\n📂 Processing Scan {scan_number}...")

    # 1. Load LAS file and info
    las = laspy.read(os.path.join(folder1, las_files[scan_number]))
    coords=np.vstack((las.x, las.y, las.z))
    point_cloud = coords.transpose()    
    r=(las.red/65535*255).astype(int)
    g=(las.green/65535*255).astype(int)
    b=(las.blue/65535*255).astype(int)
    colors = np.vstack((r,g,b)).transpose()
    points = np.vstack((las.x, las.y, las.z)).T
    center = scan_positions_df[scan_positions_df['Scan_ID'] == scan_number]
    if center.empty:
        print(f"❌ No center for scan {scan_number}")
        continue
    center_coordinates = [center.iloc[0]['X'], center.iloc[0]['Y'], center.iloc[0]['Z']]

    image, mapping = generate_spherical_image(center_coordinates, colors,point_cloud, resolution_y)
    unique_values, counts = np.unique(mapping, return_counts=True)
    num_invalid_points = np.sum(mapping == -1)
    print(f"Number of invalid points: {num_invalid_points}")
    presence_map = (mapping != -1).astype(np.uint8) * 255
    cv2.imwrite("mapping_mask.jpg", presence_map)
    cv2.imwrite('test.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    masks = mask_generator.generate(image)
    img_height = resolution_y
    img_width = 2 * resolution_y
    original_image_path = Path(original_images_folder) / original_image_files[scan_number]
    for idx, ann in enumerate(masks):
            ann['id'] = idx  # Unique ID per object

            if len(ann['bbox']) == 4:
                x_min, y_min, width, height = map(int, ann['bbox'])
                x_max = x_min + width
                y_max = y_min + height
            else:
                x_min, y_min, x_max, y_max = map(int, ann['bbox'])

            # Apply offsets
            x_min = max(0, x_min - offset_x)
            y_min = max(0, y_min - offset_y)
            x_max = min(img_width, x_max + offset_x)
            y_max = min(img_height, y_max + offset_y)

            if x_max <= x_min or y_max <= y_min:
                print(f"Warning: Skipping invalid bounding box {ann['bbox']} in {original_image_path.name}")
                continue
            
            if (x_max - x_min) < min_bbox_size or (y_max - y_min) < min_bbox_size:
                print(f"Warning: Skipping small bounding box {ann['bbox']} in {original_image_path.name}")
                continue

            cropped_object = image[y_min:y_max, x_min:x_max]

            if cropped_object.size == 0:
                print(f"Warning: Empty crop for object ID {idx} in {original_image_path.name}")
                continue

            image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_class = torch.max(output, 1)
                class_label = class_names[predicted_class.item()]

            class_names[idx] = class_label  

    segmented_image = apply_segmentation_masks(masks, image, class_names)
    cv2.iwrite('test_seg.jpg',cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    

print("\n🎉 All scans processed.")
