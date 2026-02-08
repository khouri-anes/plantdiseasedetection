import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm




def extract_leaf(image):
    # 1. Keep a copy of the ORIGINAL image for the final output
    # We must not analyze the "enhanced" pixels, only the real ones.
    original = image.copy()

    # 2. Illumination Normalization (Used ONLY for mask generation)
    # Your logic here was good for detection, just not for final analysis.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE helps separate leaf from shadow in difficult lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. Saturation Thresholding (The Pro Move)
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Otsu on Saturation is the industry standard for plant phenotyping
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological Cleanup
    # Closing fills small holes (like veins or small disease spots)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Largest Contour Extraction
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)

    # 6. Re-Draw the Mask
    # We create a new clean mask from the contour.
    # Passing -1 fills the contour, effectively fixing "White Disease" holes inside the leaf.
    leaf_mask = np.zeros_like(s)
    cv2.drawContours(leaf_mask, [largest], -1, 255, -1)

    # 7. Apply Mask to the ORIGINAL Image (Crucial Step)
    # We return the TRUE colors, not the CLAHE-modified colors.
    leaf = cv2.bitwise_and(original, original, mask=leaf_mask)

    return leaf, leaf_mask




def detect_early_blight(leaf, leaf_mask):
    # 1. Convert to HSV
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)

    # 2. Define "Healthy Green" range (The Exclusion Method)
    # Sometimes it's safer to say "Disease = Anything inside leaf that isn't Green"
    # because Early Blight has complex textures (rings).
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 3. Define the "Yellow Halo" (Chlorosis) - CRITICAL for Early Blight
    # Early Blight toxins cause yellowing around spots.
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 4. Define "Brown/Dark" Lesions (Necrosis)
    # Target browns and dark spots (low value)
    lower_brown = np.array([0, 40, 20])
    upper_brown = np.array([20, 255, 200])  # Covers red-brown to orange-brown
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # 5. Combine Disease Signs
    # Disease = Brown Spots OR Yellow Halos
    disease_candidates = cv2.bitwise_or(brown_mask, yellow_mask)

    # 6. Double Check: Remove Green parts explicitly
    # (Fixes edge cases where yellow blends into light green)
    disease_candidates = cv2.bitwise_and(disease_candidates, cv2.bitwise_not(green_mask))

    # 7. Apply Leaf Mask (Strict Boundary)
    disease_candidates = cv2.bitwise_and(disease_candidates, leaf_mask)

    # 8. Connected Component Analysis (Smarter Filtering)
    # Instead of a hard "200 pixel" limit, we use a ratio or a smaller threshold
    # to catch the "Early" spots too.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_candidates)
    disease_mask = np.zeros_like(disease_candidates)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Lower threshold to 20-50 pixels to catch starting spots.
        # But we ensure compactness (Early Blight spots are roughly circular).
        if area > 30:
            disease_mask[labels == i] = 255

    # 9. Morphological Closing to fill the "Bullseye"
    # Early Blight often has concentric rings with healthy-looking gaps.
    # Closing connects the rings into one solid lesion.
    kernel = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return disease_mask




def detect_late_blight(leaf, leaf_mask):
    # 1. Convert to HSV
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)

    # 2. Define "Healthy Green" range
    # In OpenCV HSV, Green is roughly [35, 85]
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask of everything that is GREEN
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 3. Define "Disease Color" (Browns, Yellows, Reds)
    # Late Blight is usually brown/black (Hue 0-25 or 160-180) or yellow halo (Hue 25-35)

    # Range 1: Red/Brown (Start of spectrum)
    lower_brown1 = np.array([0, 40, 50])
    upper_brown1 = np.array([35, 255, 255])

    # Range 2: Red/Purple (End of spectrum)
    lower_brown2 = np.array([160, 40, 50])
    upper_brown2 = np.array([180, 255, 255])

    # Range 3: Very Dark Necrosis (Black spots)
    # This catches the dead tissue that has no color (Hue doesn't matter, Value is low)
    # But we MUST exclude Green Hue to avoid shadows
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])  # Value < 80 is very dark

    mask_brown1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask_brown2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 4. Logic: Disease is (Brown OR Black) AND NOT Green
    disease_mask = cv2.bitwise_or(mask_brown1, mask_brown2)
    disease_mask = cv2.bitwise_or(disease_mask, mask_black)

    # CRITICAL STEP: Subtract the Green Mask explicitly
    # Even if a pixel is dark, if it is Green, it is a shadow/vein.
    disease_mask = cv2.bitwise_and(disease_mask, cv2.bitwise_not(green_mask))

    # 5. Apply the Leaf Mask (Stay inside the leaf)
    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # 6. Cleaning (Morphology)
    # Remove isolated pixels (noise)
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Close holes inside the lesion
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return disease_mask

def calculate_severity(disease_mask, leaf_mask):
    disease_pixels = np.count_nonzero(disease_mask)
    leaf_pixels = np.count_nonzero(leaf_mask)

    if leaf_pixels == 0:
        return 0.0

    return (disease_pixels / leaf_pixels) * 100


def severity_early_blight(percent):
    if percent < 0:
        return "healthy"
    elif percent < 20:
        return "mild"
    elif percent < 40:
        return "moderate"
    else:
        return "severe"

def severity_late_blight(percent):
    if percent < 0:
        return "healthy"
    elif percent < 10:
        return "mild"
    elif percent < 25:
        return "moderate"
    else:
        return "severe"


def analyze_potato_image(image_path, disease_type):
    image = cv2.imread(image_path)
    leaf, leaf_mask = extract_leaf(image)

    if leaf is None:
        return None, None

    if disease_type == "healthy":
        return "healthy", 0.0

    if disease_type == "early_blight":
        mask = detect_early_blight(leaf, leaf_mask)
        percent = calculate_severity(mask, leaf_mask)
        return severity_early_blight(percent), percent

    if disease_type == "late_blight":
        mask = detect_late_blight(leaf, leaf_mask)
        percent = calculate_severity(mask, leaf_mask)
        return severity_late_blight(percent), percent

    return None, None


def build_severity_dataset(
    plantvillage_root="PlantVillage",
    output_root="PlantVillage_Severity"
):
    os.makedirs(output_root, exist_ok=True)

    for split in ["train", "val"]:

        split_path = os.path.join(plantvillage_root, split)
        if not os.path.isdir(split_path):
            continue

        for folder in os.listdir(split_path):

            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue

            if not folder.startswith("Potato___"):
                continue

            if "healthy" in folder:
                disease_type = "healthy"
            elif "Early_blight" in folder:
                disease_type = "early_blight"
            elif "Late_blight" in folder:
                disease_type = "late_blight"
            else:
                continue

            for img in tqdm(os.listdir(folder_path), desc=f"{split}/{folder}"):
                img_path = os.path.join(folder_path, img)

                severity, percent = analyze_potato_image(img_path, disease_type)
                if severity is None:
                    continue

                target_dir = os.path.join(
                    output_root,
                    severity,
                    folder
                )
                os.makedirs(target_dir, exist_ok=True)

                shutil.copy(img_path, os.path.join(target_dir, img))

def debug_segmentation(image_path, disease_type):
    image = cv2.imread(image_path)
    leaf, leaf_mask = extract_leaf(image)

    if leaf is None:
        print("Leaf segmentation failed.")
        return

    if disease_type == "early_blight":
        disease_mask = detect_early_blight(leaf, leaf_mask)
    elif disease_type == "late_blight":
        disease_mask = detect_late_blight(leaf, leaf_mask)
    else:
        disease_mask = np.zeros_like(leaf_mask)

    # Overlay disease mask in red
    overlay = image.copy()
    overlay[disease_mask > 0] = [0, 0, 255]

    cv2.imshow("Original Image", image)
    cv2.imshow("Leaf Mask (white = leaf)", leaf_mask)
    cv2.imshow("Extracted Leaf", leaf)
    cv2.imshow("Disease Overlay (red)", overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # build_severity_dataset(plantvillage_root="PlantVillage", output_root="PlantVillage_Severity")
    debug_segmentation(
        image_path ="PlantVillage_Severity/mild/Potato___Late_blight/3f6bce87-fb66-44fb-8e76-e282af4869b9___RS_LB 3013.JPG",
        disease_type="late_blight"
    )
    # debug_segmentation(
    #     image_path="PlantVillage_Severity/moderate/Potato___Early_blight/b5127f55-debe-459a-98e5-4629750f9c38___RS_Early.B 7496.JPG",
    #     disease_type="early_blight"
    # )

