path='/home/halimo/GraduationProject/section-1-newnew/raw/B6 - F3 & S1.e57' # Path to the e57 file containing all the scans
output_dir = "PrepedScans/BestBuilding/" 

import os
import numpy
import laspy
from pye57 import E57
import numpy as np
import pandas as pd
import csv
import sys
from scipy.stats import zscore
import imageio.v3 as iio


def generate_spherical_image_from_e57(spherical_range, azimuth, elevation, colors, resolution_y=500):
    # Convert azimuth (-π to π) and elevation (-π/2 to π/2) to pixel coordinates
    resolution_x = 2 * resolution_y  # panorama aspect ratio

    x = (azimuth + np.pi) / (2 * np.pi) * resolution_x
    y = (np.pi/2 - elevation) / np.pi * resolution_y  # flip so top = +Z

    image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)
    mapping = np.full((resolution_y, resolution_x), -1, dtype=int)

    for i in range(len(spherical_range)):
        ix = np.clip(np.floor(x[i]).astype(int), 0, resolution_x - 1)
        iy = np.clip(np.floor(y[i]).astype(int), 0, resolution_y - 1)

        # Keep closest point in case of collision
        if mapping[iy, ix] == -1 or spherical_range[i] < spherical_range[mapping[iy, ix]]:
            mapping[iy, ix] = i
            image[iy, ix] = colors[i]

    return image, mapping

def main():
    limit = 10
    z_threshold = 3
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(script_dir)
    e57_path=os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),path)
    e57=E57(e57_path)
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "scan_positions.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Scan_ID", "X", "Y", "Z"])  
        for scan in range(e57.scan_count):
            try:
                position = e57.scan_position(scan)
                position = position.flatten().tolist()  
            except Exception as e:
                print(f"Error retrieving scan position for scan {scan}: {e}")
                data = e57.read_scan(scan, intensity=True, colors=True, row_column=True, transform=True, ignore_missing_fields=True)
                points = np.vstack((data['cartesianX'], data['cartesianY'], data['cartesianZ'])).T

                if points.shape[0] == 0:
                    print(f"Skipping scan {scan}: No valid points found.")
                    continue  
                position = np.mean(points, axis=0).tolist()  
                if len(position) != 3:
                    print(f"Skipping scan {scan}: Invalid computed position {position}.")
                    continue
            writer.writerow([scan] + position) 
            data = e57.read_scan(scan, intensity=True, colors=True, row_column=True, transform=True, ignore_missing_fields=True)
            points = np.vstack((data['cartesianX'], data['cartesianY'], data['cartesianZ'])).T
            distances = np.linalg.norm(points - position, axis=1)
            mask = distances <= limit
            filtered_points = points[mask]
            
            if 'colorRed' in data and 'colorGreen' in data and 'colorBlue' in data:
                colors = np.vstack((data['colorRed'], data['colorGreen'], data['colorBlue'])).T
                filtered_colors = colors[mask]

                if filtered_colors.max() > 1.0:
                    filtered_colors = filtered_colors / 255.0 
                filtered_colors = (filtered_colors * 65535).astype(np.uint16)
            else:
                filtered_colors = None
            if filtered_points.shape[0] > 0:
                z_scores = np.abs(zscore(filtered_points, axis=0))
                non_outlier_mask = (z_scores < z_threshold).all(axis=1)
                filtered_points = filtered_points[non_outlier_mask]

                if filtered_colors is not None:
                    filtered_colors = filtered_colors[non_outlier_mask]

            print(f"Scan {scan}: {points.shape[0]} points before filtering, {filtered_points.shape[0]} points within {limit} meters after filtering and after removing outliers")

            if filtered_points.shape[0] > 0:
                las_header = laspy.LasHeader(point_format=2, version="1.2")
                las = laspy.LasData(las_header)
                las.x, las.y, las.z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

                if filtered_colors is not None:
                    las.red, las.green, las.blue = filtered_colors[:, 0], filtered_colors[:, 1], filtered_colors[:, 2]

                las_path = os.path.join(output_dir, f"filtered_scan_{scan}.las")
                las.write(las_path)
                print(f"Filtered scan {scan} saved as {las_path}")
            if 'sphericalAzimuth' in data and 'sphericalElevation' in data and 'sphericalRange' in data:
                azimuth = data['sphericalAzimuth'][mask]
                elevation = data['sphericalElevation'][mask]
                spherical_range = data['sphericalRange'][mask]
                azimuth = azimuth[non_outlier_mask]
                elevation = elevation[non_outlier_mask]
                spherical_range = spherical_range[non_outlier_mask]
                spherical_img, pixel_mapping = generate_spherical_image_from_e57(
                    spherical_range, azimuth, elevation, filtered_colors)
                img_path = os.path.join(output_dir, f"spherical_image_{scan}.png")
                iio.imwrite(img_path, spherical_img)
                print(f"Spherical image saved to {img_path}")
                mapping_path = os.path.join(output_dir, f"spherical_pixel_to_index_{scan}.npy")
                np.save(mapping_path, pixel_mapping)
                print(f"Pixel index mapping saved to {mapping_path}")



    print(f"All scan positions saved to {csv_file}")

            
        
            
main()
            
