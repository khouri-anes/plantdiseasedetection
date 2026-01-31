import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm  # Professional progress bar

# --- CONFIGURATION ---
INPUT_DATASET_DIR = "PlantVillage" #  raw downloaded folder
OUTPUT_DATASET_DIR = "PlantVillage_Severity"  # Where new sorted images go

# Thresholds for severity (Percentages)
THRESH_MILD = 5.0  # 0% to 5% -> Healthy/Very Mild
THRESH_MODERATE = 15.0  # 5% to 15% -> Mild
THRESH_SEVERE = 35.0  # 15% to 35% -> Moderate, >35% -> Severe

# HSV Color Ranges (Calibrated for General Green Leaves)
# Note: In HSV, Green is roughly roughly [35, 185] in OpenCV's scale (0-179)
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])


def automated_labeling_pipeline(image_path):
    """
    Analyzes a leaf image and returns the severity ratio.
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: return None

    # 2. Preprocessing: Blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Convert to HSV Color Space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 4. Leaf Isolation (Background Removal)
    # We assume the background is distinct. We can use the Saturation channel.
    # Leaves are saturated, backgrounds (gray/black) are usually low saturation.
    # Alternatively, use Otsu's thresholding on the grayscale version.
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Healthy Tissue Segmentation
    # Create a mask for what is "Green"
    healthy_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Refine the healthy mask: Only consider green pixels INSIDE the leaf area
    healthy_mask = cv2.bitwise_and(healthy_mask, leaf_mask)

    # 6. Disease Segmentation
    # Logic: Disease = (Leaf Area) - (Healthy Green Area)
    # We use bitwise XOR or subtraction.
    disease_mask = cv2.subtract(leaf_mask, healthy_mask)

    # 7. Morphological Cleanup (The Pro Touch)
    # Remove small white noise specks from the disease mask
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 8. Quantification
    total_pixels = np.count_nonzero(leaf_mask)
    disease_pixels = np.count_nonzero(disease_mask)

    if total_pixels == 0:
        return 0.0  # Avoid division by zero

    severity_ratio = (disease_pixels / total_pixels) * 100

    return severity_ratio


def organize_dataset():
    """
    Iterates through raw folders and reorganizes images based on severity.
    """
    # Create output directories
    categories = ['Healthy', 'Mild', 'Moderate', 'Severe']
    for cat in categories:
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, cat), exist_ok=True)

    print(f"🚀 Starting automated relabeling pipeline...")

    # Walk through the raw dataset
    processed_count = 0

    for root, dirs, files in os.walk(INPUT_DATASET_DIR):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(root, file)

            # --- VISION ANALYSIS ---
            ratio = automated_labeling_pipeline(img_path)

            if ratio is None: continue

            # --- CLASSIFICATION LOGIC ---
            label = ""
            if ratio < THRESH_MILD:
                label = 'Healthy'
            elif ratio < THRESH_MODERATE:
                label = 'Mild'
            elif ratio < THRESH_SEVERE:
                label = 'Moderate'
            else:
                label = 'Severe'

            # --- MOVE / COPY FILE ---
            # We rename the file to include the ratio for debugging (e.g., leaf_1_12.5pct.jpg)
            new_filename = f"{os.path.splitext(file)[0]}_{ratio:.1f}pct.jpg"
            dest_path = os.path.join(OUTPUT_DATASET_DIR, label, new_filename)

            shutil.copy2(img_path, dest_path)
            processed_count += 1

    print(f"\n✅ Processing Complete.")
    print(f"Total images relabeled: {processed_count}")
    print(f"Output saved to: {OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    organize_dataset()