import cv2
import numpy as np
import os
import shutil
import csv
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

INPUT_ROOT = "Datasets/dataset_yolo/images"
OUTPUT_ROOT = "Datasets/PlantVillage_YOLO_SEG_FINAL"

# Set True to visually check masks in 'OUTPUT_ROOT/debug'
DEBUG_SAVE = True

# CRITICAL UPDATE: UNIQUE CLASS IDs
# Each Crop+Disease combo gets a unique ID so the model knows the crop.
CLASS_MAP = {
    # --- PEPPER (0-1) ---
    "Pepper__bell___Bacterial_spot": 0,
    "Pepper__bell___healthy": -1,

    # --- POTATO (1-2) ---
    "Potato___Early_blight": 1,
    "Potato___Late_blight": 2,
    "Potato___healthy": -1,

    # --- TOMATO (3-11) ---
    "Tomato_Bacterial_spot": 3,
    "Tomato_Early_blight": 4,
    "Tomato_Late_blight": 5,
    "Tomato_Leaf_Mold": 6,
    "Tomato_Septoria_leaf_spot": 7,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 8,
    "Tomato__Target_Spot": 9,
    "Tomato__Tomato_mosaic_virus": 10,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 11,
    "Tomato_healthy": -1,
}


# =========================================================
# STEP 1: LEAF SEGMENTATION (GRABCUT)
# =========================================================

def extract_leaf_grabcut(image):
    """
    Extracts the leaf from background using GrabCut.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (20, 20, w - 40, h - 40)  # Margin to avoid borders

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except:
        return None, None

    # Foreground (1) + Probable Foreground (3)
    leaf_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(leaf_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    leaf = cv2.bitwise_and(image, image, mask=clean_mask)
    return leaf, clean_mask


# =========================================================
# UTILITIES & FILTERS
# =========================================================

def remove_green(hsv, mask):
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    green = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(mask, cv2.bitwise_not(green))


def texture_filter(gray, thresh_ratio=0.04, min_thresh=20):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap).astype(np.uint8)
    thresh = max(min_thresh, int(thresh_ratio * np.mean(texture)))
    _, mask = cv2.threshold(texture, thresh, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask, leaf_mask, min_area_ratio=0.0005):
    mask = cv2.bitwise_and(mask, leaf_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

    min_area = min_area_ratio * np.count_nonzero(leaf_mask)
    final = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(final, [c], -1, 255, -1)
    return final

def grabcut_leaf_segmentation(image):
    """
    Extracts the leaf from background using GrabCut.
    This is robust against yellow, brown, and dark leaves.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Assume leaf is roughly centered, ignore image borders
    rect = (20, 20, w - 40, h - 40)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            rect,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_RECT
        )
    except:
        return None, None

    # Foreground mask: sure + probable foreground
    leaf_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # Keep only the largest connected component (the leaf)
    contours, _ = cv2.findContours(
        leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(leaf_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    leaf = cv2.bitwise_and(image, image, mask=clean_mask)

    return leaf, clean_mask
# =========================================================
# DISEASE LOGIC
# =========================================================

def early_blight_mask(hsv, gray, leaf_mask):
    yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
    brown = cv2.inRange(hsv, (0, 40, 20), (22, 255, 200))
    base = cv2.bitwise_or(yellow, brown)
    base = remove_green(hsv, base)
    texture = texture_filter(gray)
    return clean_mask(cv2.bitwise_and(base, texture), leaf_mask)


def late_blight_mask(hsv, gray, leaf_mask):
    brown1 = cv2.inRange(hsv, (0, 40, 50), (30, 255, 255))
    brown2 = cv2.inRange(hsv, (160, 40, 50), (180, 255, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
    base = cv2.bitwise_or(brown1, brown2)
    base = cv2.bitwise_or(base, black)
    base = remove_green(hsv, base)
    texture = texture_filter(gray, thresh_ratio=0.03)
    # Late blight necrosis (black) can be smooth, so we keep 'black' even without texture
    combined = cv2.bitwise_or(cv2.bitwise_and(base, texture), black)
    return clean_mask(combined, leaf_mask)


def septoria_mask(hsv, gray, leaf_mask):
    # Brown spots + Gray Centers
    brown = cv2.inRange(hsv, (0, 40, 50), (25, 255, 200))
    lower_gray = np.array([0, 0, 100]);
    upper_gray = np.array([180, 30, 255])
    gray_center = cv2.bitwise_and(cv2.inRange(hsv, lower_gray, upper_gray), leaf_mask)

    base = cv2.bitwise_or(brown, gray_center)
    base = remove_green(hsv, base)
    texture = texture_filter(gray, thresh_ratio=0.05)
    return clean_mask(cv2.bitwise_and(base, texture), leaf_mask, min_area_ratio=0.0002)


def bacterial_spot_mask(hsv, gray, leaf_mask):
    dark = cv2.inRange(hsv, (0, 40, 40), (30, 255, 150))
    base = remove_green(hsv, dark)
    texture = texture_filter(gray, thresh_ratio=0.06)
    return clean_mask(cv2.bitwise_and(base, texture), leaf_mask)


def leaf_mold_mask(hsv, gray, leaf_mask):
    grayish = cv2.inRange(hsv, (0, 0, 60), (180, 60, 160))
    base = remove_green(hsv, grayish)
    texture = texture_filter(gray, thresh_ratio=0.03)
    return clean_mask(cv2.bitwise_and(base, texture), leaf_mask)


def spider_mite_mask(hsv, gray, leaf_mask):
    stipple = cv2.inRange(hsv, (15, 30, 120), (35, 150, 255))
    texture = texture_filter(gray, thresh_ratio=0.07)
    return clean_mask(cv2.bitwise_and(stipple, texture), leaf_mask, min_area_ratio=0.0002)


def target_spot_mask(hsv, gray, leaf_mask):
    brown = cv2.inRange(hsv, (10, 40, 40), (30, 255, 200))
    base = remove_green(hsv, brown)
    texture = texture_filter(gray)
    return clean_mask(cv2.bitwise_and(base, texture), leaf_mask)


def mosaic_virus_mask(hsv, gray, leaf_mask):
    lower = np.array([20, 40, 40]);
    upper = np.array([45, 255, 255])
    color_shift = cv2.inRange(hsv, lower, upper)
    texture = texture_filter(gray, thresh_ratio=0.03)
    return clean_mask(cv2.bitwise_or(color_shift, texture), leaf_mask, min_area_ratio=0.001)


def curl_virus_mask(hsv, gray, leaf_mask):
    # Structural disease -> Return Full Leaf
    return leaf_mask


# =========================================================
# DISPATCHER
# =========================================================
DISEASE_RULES = {
    "early_blight": early_blight_mask,
    "late_blight": late_blight_mask,
    "septoria": septoria_mask,
    "bacterial_spot": bacterial_spot_mask,
    "leaf_mold": leaf_mold_mask,
    "spider_mites": spider_mite_mask,
    "target_spot": target_spot_mask,
    "mosaic": mosaic_virus_mask,
    "curl": curl_virus_mask
}


def get_lesion_mask(leaf, leaf_mask, disease_folder_name):
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
    name = disease_folder_name.lower()

    for key, func in DISEASE_RULES.items():
        if key in name:
            return func(hsv, gray, leaf_mask)

    return np.zeros_like(leaf_mask)  # Fallback / Healthy


# =========================================================
# YOLO UTILS
# =========================================================

def mask_to_yolo_polygons(mask, class_id, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        if len(cnt) < 3: continue

        poly = []
        for point in cnt.reshape(-1, 2):
            # Normalize and clamp to 0-1 to ensure YOLO compatibility
            x = max(0, min(1, point[0] / img_w))
            y = max(0, min(1, point[1] / img_h))
            poly.append(f"{x:.6f} {y:.6f}")

        labels.append(f"{class_id} " + " ".join(poly))
    return labels


def save_debug_visuals(image, leaf_mask, lesion_mask, disease, img_name, split):
    debug_dir = os.path.join(OUTPUT_ROOT, "debug", split, disease)
    os.makedirs(debug_dir, exist_ok=True)

    overlay = image.copy()
    # Green Leaf
    overlay[leaf_mask > 0] = [0, 255, 0]
    # Red Lesions
    overlay[lesion_mask > 0] = [0, 0, 255]

    # Text
    cv2.putText(overlay, disease, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, img_name), overlay)


# =========================================================
# MAIN PIPELINE
# =========================================================

def build_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    metadata_rows = []

    print(f"🚀 Starting Dataset Generation in: {OUTPUT_ROOT}")

    for split in ["train", "val"]:
        # Create directories
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

        split_path = os.path.join(INPUT_ROOT, split)
        if not os.path.exists(split_path): continue

        for disease in os.listdir(split_path):
            if disease not in CLASS_MAP: continue

            class_id = CLASS_MAP[disease]
            folder = os.path.join(split_path, disease)

            print(f"📂 Processing {split}/{disease}...")

            for img_name in tqdm(os.listdir(folder)):
                img_path = os.path.join(folder, img_name)
                image = cv2.imread(img_path)
                if image is None: continue

                h, w = image.shape[:2]

                # 1. GrabCut Leaf
                leaf, leaf_mask = extract_leaf_grabcut(image)
                if leaf is None: continue  # Skip if segmentation failed

                labels = []
                severity = 0.0
                lesion_mask = None  # Initialize to avoid crash

                # 2. Disease Detection (if not healthy)
                if class_id != -1:
                    lesion_mask = get_lesion_mask(leaf, leaf_mask, disease)

                    leaf_area = np.count_nonzero(leaf_mask)
                    lesion_area = np.count_nonzero(lesion_mask)

                    if leaf_area > 0:
                        if "curl" in disease.lower():
                            severity = 1.0  # Structural = 100%
                        else:
                            severity = lesion_area / leaf_area

                    # Only write labels if severity > 1% (filters noise)
                    if severity >= 0.01:
                        labels = mask_to_yolo_polygons(lesion_mask, class_id, w, h)

                # 3. Save Debug (Only if lesion_mask exists)
                if DEBUG_SAVE and lesion_mask is not None:
                    save_debug_visuals(image, leaf_mask, lesion_mask, disease, img_name, split)

                # 4. Save Image & Label
                shutil.copy(img_path, os.path.join(OUTPUT_ROOT, "images", split, img_name))

                label_path = os.path.join(OUTPUT_ROOT, "labels", split, os.path.splitext(img_name)[0] + ".txt")
                with open(label_path, "w") as f:
                    if labels:
                        f.write("\n".join(labels))

                metadata_rows.append([img_name, disease, round(severity, 4), split])

    # Save CSV
    with open(os.path.join(OUTPUT_ROOT, "metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_name", "severity", "split"])
        writer.writerows(metadata_rows)

    print("\n✅ DONE! Dataset is ready.")




def debug_segmentation(image_path, disease_folder_name):
    """
    Visual debugging tool for:
    - GrabCut leaf segmentation
    - Disease lesion detection (weak labels)

    Green  = Leaf
    Red    = Detected lesions
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image.")
        return

    # --- Step 1: Leaf Segmentation ---
    leaf, leaf_mask = extract_leaf_grabcut(image)
    leaf2,leaf_mask2=grabcut_leaf_segmentation(image)
    if leaf is None:
        print("❌ GrabCut failed.")
        return

    # --- Step 2: Lesion Detection ---
    lesion_mask = get_lesion_mask(
        leaf=leaf,
        leaf_mask=leaf_mask,
        disease_folder_name=disease_folder_name
    )
    lesion_mask2 = get_lesion_mask(
        leaf=leaf2,
        leaf_mask=leaf_mask2,
        disease_folder_name=disease_folder_name
    )

    # --- Step 3: Visualization ---
    overlay = image.copy()
    overlay2 = image.copy()

    # Leaf = Green
    overlay[leaf_mask > 0] = [0, 255, 0]
    overlay2[leaf_mask2 > 0] = [0, 255, 0]

    # Lesions = Red (overwrites green where needed)
    overlay[lesion_mask > 0] = [0, 0, 255]
    overlay2[lesion_mask2 > 0] = [0, 0, 255]

    # --- Display ---
    cv2.imshow("Original Image", image)
    cv2.imshow("Leaf  (GrabCut)", leaf)
    cv2.imshow("Leaf Mask (GrabCut)", leaf_mask)

    cv2.imshow("Lesion Mask (Weak Labels)", lesion_mask)
    cv2.imshow("Overlay (Green=Leaf, Red=Lesions)", overlay)
    cv2.imshow("Leaf Mask 2(GrabCut)", leaf_mask2)
    cv2.imshow("Leaf  2(GrabCut)", leaf2)
    cv2.imshow("Lesion Mask2 (Weak Labels)", lesion_mask2)
    cv2.imshow("Overlay2 (Green=Leaf, Red=Lesions)", overlay2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Uncomment this to run the full process
    # build_dataset()

    # Uncomment this to test a single image
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Spider_mites_Two_spotted_spider_mite/d75573fb-2bd8-43fa-9af5-3f219e35142f___RS_Late.B 5070.jpg",
    #     disease_folder_name="Tomato_Spider_mites_Two_spotted_spider_mite"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Late_blight/2d63d91d-5366-4f88-83e1-cc5ce4b16d2e___GHLB2 Leaf 156.1.jpg",
    #     disease_folder_name="Tomato_Late_blight"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Leaf_Mold/ff01be71-2cc9-4e83-b907-81c21d31983b___Crnl_L.Mold 9117.jpg",
    #     disease_folder_name="Tomato_Leaf_Mold"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Septoria_leaf_spot/0bcb24a9-bf45-4008-b9df-1c729f977b17___Matt.S_CG 7835.JPG",
    #     disease_folder_name="Tomato_Septoria_leaf_spot"
    # )
    # debug_segmentation(
    #         image_path="Datasets/dataset_yolo/images/train/Tomato__Target_Spot/2c93d540-be61-4546-9cbc-74c776f19379___Com.G_TgS_FL 0690.JPG",
    #         disease_folder_name="Tomato__Target_Spot"
    #     )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato__Tomato_mosaic_virus/65226438-efaa-4bf4-91e7-e459e3528565___PSU_CG 2143.JPG",
    #     disease_folder_name="Tomato__Tomato_mosaic_virus"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato__Tomato_YellowLeaf__Curl_Virus/4c60c0aa-ae90-4a67-93d9-715ef660e53a___UF.GRC_YLCV_Lab 03026.JPG",
    #     disease_folder_name="Tomato__Tomato_YellowLeaf__Curl_Virus"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Pepper__bell___Bacterial_spot/0a4c007d-41ab-4659-99cb-8a4ae4d07a55___NREC_B.Spot 1954.JPG",
    #     disease_folder_name="Pepper__bell___Bacterial_spot"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Potato___Early_blight/34f44be4-9dc5-460b-896d-cd6593166fe7___RS_Early.B 8175.JPG",
    #     disease_folder_name="Potato___Early_blight"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Potato___Late_blight/d7fdf2fa-8a55-4ff5-86a2-ca73c5883e93___RS_LB 4665.JPG",
    #     disease_folder_name="Potato___Late_blight"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Bacterial_spot/177b8ce1-d77a-41d7-93ac-852d37d69faa___GCREC_Bact.Sp 3390.JPG",
    #     disease_folder_name="Tomato_Bacterial_spot"
    # )
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Early_blight/250df77c-cb57-4e85-9cf2-531033930030___RS_Erly.B 6477.JPG",
    #     disease_folder_name="Tomato_Early_blight"
    # )
    debug_segmentation(
        image_path="Datasets/dataset_yolo/images/train/Tomato_Early_blight/250df77c-cb57-4e85-9cf2-531033930030___RS_Erly.B 6477.JPG",
        disease_folder_name="Tomato_Early_blight"
    )

