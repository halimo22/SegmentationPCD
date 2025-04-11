import os
import re
import cv2
import numpy as np
import pandas as pd
import laspy
from PIL import Image
from pathlib import Path
import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.models import efficientnet_b7
from util import generate_spherical_image, apply_segmentation_masks
from classifier import EfficientNetB7Classifier

MODEL_PATH = "Models/sam_vit_h_4b8939.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.80
)

model = EfficientNetB7Classifier().to(DEVICE)
model.load_state_dict(torch.load("Models/EffiecientNetB7/best_model.pth", map_location=DEVICE))
model.eval()
class_names = model.class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_file = "PrepedScans/BestBuilding/scan_positions.csv"
folder1 = "PrepedScans/BestBuilding"
output_folder_seg = Path("SegmentedImages3")
output_folder_csv = Path("ClassifiedPoints")
output_folder_seg.mkdir(exist_ok=True)
output_folder_csv.mkdir(exist_ok=True)

offset_x, offset_y = 10, 10
min_bbox_size = 20
resolution_y = 500

scan_positions_df = pd.read_csv(csv_file)
scan_positions_df['Scan_ID'] = scan_positions_df['Scan_ID'].astype(int)

las_files = {}
for f in os.listdir(folder1):
    if f.endswith(".las"):
        match = re.search(r'(\d+)', f)
        if match:
            scan_id = int(match.group(1))
            las_files[scan_id] = f
            
def process_scan(scan_number):
    las = laspy.read(os.path.join(folder1, las_files[scan_number]))
    point_cloud = np.vstack((las.x, las.y, las.z)).T
    r = (las.red / 65535 * 255).astype(int)
    g = (las.green / 65535 * 255).astype(int)
    b = (las.blue / 65535 * 255).astype(int)
    colors = np.vstack((r, g, b)).T
    center = scan_positions_df[scan_positions_df['Scan_ID'] == scan_number]
    if center.empty:
        return
    center_coordinates = [center.iloc[0]['X'], center.iloc[0]['Y'], center.iloc[0]['Z']]
    image, mapping = generate_spherical_image(center_coordinates, colors, point_cloud, resolution_y)
    presence_map = (mapping != -1).astype(np.uint8) * 255
    cv2.imwrite(f"mapping_mask_scan_{scan_number}.jpg", presence_map)
    cv2.imwrite(f'test_scan_{scan_number}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    masks = mask_generator.generate(image)
    img_height, img_width = resolution_y, 2 * resolution_y
    class_labels = {}
    id_to_pixels = {}
    for idx, ann in enumerate(masks):
        ann['id'] = idx
        bbox = ann['bbox']
        if len(bbox) == 4:
            x_min, y_min, width, height = map(int, bbox)
            x_max = x_min + width
            y_max = y_min + height
        else:
            x_min, y_min, x_max, y_max = map(int, bbox)
        x_min = max(0, x_min - offset_x)
        y_min = max(0, y_min - offset_y)
        x_max = min(img_width, x_max + offset_x)
        y_max = min(img_height, y_max + offset_y)
        if x_max <= x_min or y_max <= y_min:
            continue
        if (x_max - x_min) < min_bbox_size or (y_max - y_min) < min_bbox_size:
            continue
        cropped_object = image[y_min:y_max, x_min:x_max]
        if cropped_object.size == 0:
            continue
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB)).convert("RGB")
        image_tensor = transform(cropped_pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            class_label = class_names[predicted_class.item()]
        class_labels[idx] = class_label
        mask = ann['segmentation']
        pixels = np.argwhere(mask)
        if idx in id_to_pixels:
            id_to_pixels[idx] = np.vstack((id_to_pixels[idx], pixels))
        else:
            id_to_pixels[idx] = pixels
    segmented_image = apply_segmentation_masks(masks, image, class_labels)
    cv2.imwrite(str(output_folder_seg / f"test_seg_scan_{scan_number}.jpg"),
                cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    classified_points = []
    for obj_id, pixels in id_to_pixels.items():
        label = class_labels.get(obj_id, "Unknown")
        for y, x in pixels:
            if 0 <= y < mapping.shape[0] and 0 <= x < mapping.shape[1]:
                point_idx = mapping[y, x]
                if point_idx == -1:
                    continue
                x_, y_, z_ = point_cloud[point_idx]
                r_, g_, b_ = colors[point_idx]
                classified_points.append([x_, y_, z_, r_, g_, b_, label, obj_id])
    df = pd.DataFrame(classified_points, columns=[
        "x", "y", "z", "r", "g", "b", "class", "instance_id"
    ])
    df.to_csv(output_folder_csv / f"scan_{scan_number}_classified.csv", index=False, float_format="%.6f")
    
if __name__ == "__main__":
    for scan_number in tqdm(las_files.keys(), desc="Processing Scans"):
        process_scan(scan_number)
    print("\n🎉 All scans processed.")
