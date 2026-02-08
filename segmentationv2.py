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
OUTPUT_ROOT = "Datasets/PlantVillage_YOLO_SEG"

DEBUG_SAVE = False  # Set True to save overlay images for inspection

# Class mapping (YOLO class indices)
# Healthy images get NO lesions (empty label file)
CLASS_MAP = {
    "Potato___Early_blight": 0,
    "Potato___Late_blight": 1,
    "Potato___healthy": -1
}

# =========================================================
# STEP 1: LEAF SEGMENTATION (GRABCUT)
# =========================================================

def extract_leaf_grabcut(image):
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
# STEP 2: DISEASE-SPECIFIC LESION SEGMENTATION (WEAK LABELS)
# =========================================================
def get_lesion_mask(leaf, leaf_mask, disease_folder_name):
    """
    Génère un masque binaire des lésions basé sur la couleur et la texture.
    C'est une "Supervision Faible" (Weak Supervision) pour entraîner l'IA.
    """
    # Conversion Espaces Couleur
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

    disease_mask = np.zeros_like(leaf_mask)
    name = disease_folder_name.lower()

    # --- 1. LOGIQUE COULEUR PAR MALADIE ---
    if "early" in name:
        # Early Blight : Taches brunes + Halo jaune
        yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
        brown = cv2.inRange(hsv, (0, 40, 20), (22, 255, 200))
        disease_mask = cv2.bitwise_or(yellow, brown)

    elif "late" in name:
        # Late Blight : Nécrose sombre / noire
        brown1 = cv2.inRange(hsv, (0, 40, 50), (30, 255, 255))
        brown2 = cv2.inRange(hsv, (160, 40, 50), (180, 255, 255))
        # Noir profond (nécrose morte)
        black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))

        disease_mask = cv2.bitwise_or(brown1, brown2)
        disease_mask = cv2.bitwise_or(disease_mask, black)

    else:
        # Sain ou autre
        return np.zeros_like(leaf_mask)

    # --- 2. EXCLUSION DU VERT (CRITIQUE) ---
    # On retire tout ce qui est "Vert Sain" pour éviter les faux positifs sur les veines
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    disease_mask = cv2.bitwise_and(disease_mask, cv2.bitwise_not(green_mask))

    # --- 3. FILTRE DE TEXTURE ADAPTATIF ---
    # Les lésions sont rugueuses, la feuille saine est lisse.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap).astype(np.uint8)

    # Seuil dynamique : s'adapte au flou de l'image
    adaptive_thresh = max(20, int(0.04 * np.mean(texture)))
    _, texture_mask = cv2.threshold(texture, adaptive_thresh, 255, cv2.THRESH_BINARY)

    # La maladie doit être (Bonne Couleur ET Texture Rugueuse) OU (Très Sombre/Mort)
    color_and_texture = cv2.bitwise_and(disease_mask, texture_mask)

    # Exception pour les taches noires lisses (nécrose ancienne)
    dark_spots = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    dark_spots = cv2.bitwise_and(dark_spots, cv2.bitwise_not(green_mask))

    candidates = cv2.bitwise_or(color_and_texture, dark_spots)

    # --- 4. CONTRAINTES FINALES ---
    # On reste dans la feuille
    candidates = cv2.bitwise_and(candidates, leaf_mask)

    # Nettoyage morphologique (boucher les trous, virer le bruit)
    kernel = np.ones((5, 5), np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 5. FILTRE DE SURFACE MINIMALE ---
    # On ignore les taches minuscules (< 0.05% de la feuille)
    min_area = 0.0005 * np.count_nonzero(leaf_mask)
    final_mask = np.zeros_like(candidates)

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask

# =========================================================
# STEP 3: MASK → YOLOv8 SEG POLYGONS
# =========================================================

def mask_to_yolo_polygons(mask, class_id, img_w, img_h):
    """
    Converts a binary mask into YOLOv8-Seg polygon format.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    labels = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        # Polygon simplification (critical for stability)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)

        if len(cnt) < 3:
            continue

        cnt = cnt.reshape(-1, 2)
        poly = []

        for x, y in cnt:
            poly.append(f"{x / img_w:.6f} {y / img_h:.6f}")

        labels.append(f"{class_id} " + " ".join(poly))

    return labels


# =========================================================
# debug
# =========================================================

def save_debug_visuals(
    image,
    leaf_mask,
    lesion_mask,
    disease,
    img_name,
    split
):
    """
    Saves overlay images to visually debug segmentation quality.
    """
    debug_dir = os.path.join(
        OUTPUT_ROOT, "debug", split, disease
    )
    os.makedirs(debug_dir, exist_ok=True)

    overlay = image.copy()

    # Leaf mask overlay (BLUE)
    leaf_col = np.zeros_like(image)
    leaf_col[:, :, 0] = leaf_mask
    overlay = cv2.addWeighted(overlay, 1.0, leaf_col, 0.3, 0)

    # Lesion mask overlay (RED)
    lesion_col = np.zeros_like(image)
    lesion_col[:, :, 2] = lesion_mask
    overlay = cv2.addWeighted(overlay, 1.0, lesion_col, 0.6, 0)

    # Add label text
    cv2.putText(
        overlay,
        disease,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out_path = os.path.join(debug_dir, img_name)
    cv2.imwrite(out_path, overlay)

# =========================================================
# STEP 4: DATASET PIPELINE + METADATA
# =========================================================

def build_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

    metadata_path = os.path.join(OUTPUT_ROOT, "metadata.csv")
    metadata_rows = []

    print("🚀 Building weakly-supervised YOLOv8-Seg dataset...")

    for split in ["train", "val"]:
        split_path = os.path.join(INPUT_ROOT, split)
        if not os.path.exists(split_path):
            continue

        for disease in os.listdir(split_path):
            if disease not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[disease]
            folder = os.path.join(split_path, disease)

            for img_name in tqdm(os.listdir(folder), desc=f"{split}/{disease}"):
                img_path = os.path.join(folder, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                h, w = image.shape[:2]

                leaf, leaf_mask = extract_leaf_grabcut(image)
                if leaf is None:
                    continue

                labels = []
                severity = 0.0

                if class_id != -1:
                    lesion_mask = get_lesion_mask(leaf, leaf_mask, disease)

                    leaf_area = np.count_nonzero(leaf_mask)
                    lesion_area = np.count_nonzero(lesion_mask)

                    if leaf_area > 0:
                        severity = lesion_area / leaf_area

                    if DEBUG_SAVE:
                        save_debug_visuals(
                            image=image,
                            leaf_mask=leaf_mask,
                            lesion_mask=lesion_mask,
                            disease=disease,
                            img_name=img_name,
                            split=split
                        )

                    # Skip extremely weak/noisy lesions
                    if severity >= 0.01:
                        labels = mask_to_yolo_polygons(
                            lesion_mask, class_id, w, h
                        )

                # Copy image
                shutil.copy(
                    img_path,
                    os.path.join(OUTPUT_ROOT, "images", split, img_name)
                )

                # Write label file (empty is valid)
                label_path = os.path.join(
                    OUTPUT_ROOT,
                    "labels",
                    split,
                    os.path.splitext(img_name)[0] + ".txt"
                )
                with open(label_path, "w") as f:
                    if labels:
                        f.write("\n".join(labels))

                # Metadata row
                metadata_rows.append([
                    img_name,
                    disease,
                    round(severity, 4),
                    "weak-label",
                    split
                ])

    # Write metadata CSV
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_name", "severity", "label_type", "split"])
        writer.writerows(metadata_rows)

    print("✅ Dataset generation completed successfully.")
    print(f"📂 Output: {OUTPUT_ROOT}")






# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    build_dataset()
