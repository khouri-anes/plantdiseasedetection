import cv2
import numpy as np
import os
import shutil
import csv
from tqdm import tqdm

# CONFIGURATION

INPUT_ROOT = "Datasets/dataset_yolo/images"
OUTPUT_ROOT = "Datasets/PlantVillage_YOLO_SEG_FINAL"

# Set True to visually check masks in 'OUTPUT_ROOT/debug'
DEBUG_SAVE = True


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


# STEP 1: LEAF SEGMENTATION



def extract_leaf_mask(image):
    """
    Isolates the leaf from the background.
    Returns the binary mask of the leaf and the total leaf area.
    """
    hsv_image=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    LOWER_LEAF = np.array([15, 40, 40])
    UPPER_LEAF = np.array([95, 255, 255])

    # Spot detection thresholds


    KERNEL = np.ones((5, 5), np.uint8)
    # 1. Create a mask based on color range
    mask = cv2.inRange(hsv_image, LOWER_LEAF, UPPER_LEAF)

    # 2. Clean up noise using Morphological Closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    # 3. Find contours to identify distinct objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0

    # 4. Keep only the largest contour (assuming it is the leaf)
    largest = max(contours, key=cv2.contourArea)

    # 5. Draw a clean mask of just the largest contour
    final_leaf_mask = np.zeros_like(mask)
    cv2.drawContours(final_leaf_mask, [largest], -1, 255, -1)

    leaf_area = cv2.countNonZero(final_leaf_mask)
    leaf = cv2.bitwise_and(image, image, mask=final_leaf_mask)
    return leaf, final_leaf_mask
#
# def extract_leaf_grabcut(image):
#     """
#     Extracts the leaf from background using GrabCut.
#     """
#     h, w = image.shape[:2]
#     mask = np.zeros((h, w), np.uint8)
#     rect = (20, 20, w - 40, h - 40)  # Margin to avoid borders
#
#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)
#
#     try:
#         cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#     except:
#         return None, None
#
#     # Foreground (1) + Probable Foreground (3)
#     leaf_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
#
#     contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None
#
#     largest = max(contours, key=cv2.contourArea)
#     clean_mask = np.zeros_like(leaf_mask)
#     cv2.drawContours(clean_mask, [largest], -1, 255, -1)
#
#     leaf = cv2.bitwise_and(image, image, mask=clean_mask)
#     return leaf, clean_mask


# UTILITIES & FILTERS

def remove_green(hsv, mask):
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    green = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(mask, cv2.bitwise_not(green))


def texture_filter(gray, thresh_ratio=0.035, min_thresh=15):
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    texture = lap + 0.5 * (sobelx + sobely)
    texture = texture.astype(np.uint8)

    thresh = max(min_thresh, int(thresh_ratio * np.percentile(texture, 75)))
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




def local_contrast_mask(gray, window=15, min_std=6):
    """
    Keeps regions with sufficient local intensity variation.
    Shadows tend to have low local std deviation.
    """
    mean = cv2.blur(gray, (window, window))
    sqr_mean = cv2.blur(gray**2, (window, window))
    std = np.sqrt(sqr_mean - mean**2)

    mask = (std > min_std).astype(np.uint8) * 255
    return mask



def dark_lesion_mask(hsv):
    """
    Detects dark gray / desaturated early blight lesions
    """
    # Low saturation, low value (gray & dark)
    lower = np.array([0, 0, 20])
    upper = np.array([180, 50, 90])
    dark = cv2.inRange(hsv, lower, upper)

    # Remove green regions just in case
    dark = remove_green(hsv, dark)
    return dark
def necrotic_black_mask(hsv):
    """
    Detects true necrotic (late blight) tissue:
    - dark
    - low saturation
    - NOT green
    """
    # Dark + desaturated
    lower = np.array([0, 0, 20])
    upper = np.array([180, 60, 100])
    black = cv2.inRange(hsv, lower, upper)

    # Explicitly remove green hues
    black = remove_green(hsv, black)

    return black
def dark_lesion_validator(gray, min_std=5, window=21):
    mean = cv2.blur(gray, (window, window))
    sqr_mean = cv2.blur(gray**2, (window, window))
    std = np.sqrt(sqr_mean - mean**2)
    return (std > min_std).astype(np.uint8) * 255

from skimage.feature import local_binary_pattern

def lbp_map(gray, radius=2, points=16):
    """
    Computes LBP texture map
    """
    lbp = local_binary_pattern(
        gray,
        P=points,
        R=radius,
        method='uniform'
    )
    return lbp


def lbp_variance_mask(lbp, leaf_mask, window=15, k=1.5):
    mean = cv2.blur(lbp, (window, window))
    sqr_mean = cv2.blur(lbp**2, (window, window))
    var = sqr_mean - mean**2

    # Normalize inside leaf only
    vals = var[leaf_mask > 0]
    mu, sigma = np.mean(vals), np.std(vals)

    thresh = mu + k * sigma
    mask = (var > thresh).astype(np.uint8) * 255
    return mask

def suppress_veins(gray):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) + \
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.abs(sobel).astype(np.uint8)

    _, veins = cv2.threshold(sobel, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    veins = cv2.morphologyEx(veins, cv2.MORPH_CLOSE, kernel)

    return cv2.bitwise_not(veins)
def mosaic_sanity(mask, leaf_mask, max_ratio=0.45):
    ratio = np.count_nonzero(mask) / np.count_nonzero(leaf_mask)
    if ratio > max_ratio:
        return np.zeros_like(mask)
    return mask

def green_mask(hsv):
    return cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

# =========================================================
# DISEASE LOGIC
# =========================================================



def early_blight_mask(leaf,hsv, gray, leaf_mask):
    yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
    brown  = cv2.inRange(hsv, (0, 40, 20), (25, 255, 200))
    dark   = dark_lesion_mask(hsv)

    base = cv2.bitwise_or(yellow, brown)
    base = cv2.bitwise_or(base, dark)
    base = remove_green(hsv, base)

    texture = texture_filter(gray)
    contrast = local_contrast_mask(gray)

    combined = cv2.bitwise_and(base, texture)
    combined = cv2.bitwise_and(combined, contrast)

    return clean_mask(combined, leaf_mask)



def late_blight_mask(leaf,hsv, gray, leaf_mask):
    brown1 = cv2.inRange(hsv, (0, 40, 40), (30, 255, 200))
    brown2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 200))

    brown = cv2.bitwise_or(brown1, brown2)
    brown = remove_green(hsv, brown)

    black = necrotic_black_mask(hsv)

    texture = texture_filter(gray, thresh_ratio=0.03)
    # contrast = dark_lesion_validator(gray)

    # Brown must have texture
    brown_part = cv2.bitwise_and(brown, texture)

    # # Black must have some local variation
    # black_part = cv2.bitwise_and(black, contrast)

    # combined = cv2.bitwise_or(brown_part, black_part)
    combined = cv2.bitwise_or(brown_part, black)
    return clean_mask(combined, leaf_mask)


def leaf_mold_mask(leaf,hsv,gray,mask):
    l=cv2.inRange(hsv,(15,40,40),(45,255,255))
    return cv2.bitwise_and(l,mask)

def spider_mite_mask(leaf,hsv, gray, leaf_mask):

    # Spider Mites :
    # La lésion est définie comme une zone qui est À LA FOIS décolorée ET texturée.

    # 1. Color Detection (Chlorosis)

    yellows = cv2.inRange(hsv, (10, 30, 100), (35, 255, 255))
    h, s, v = cv2.split(hsv)

    # Very bright
    bright = cv2.inRange(v, 200, 255)

    # Very low saturation
    low_sat = cv2.inRange(s, 0, 40)

    white_pixels = cv2.bitwise_and(bright, low_sat)

    # Keep only inside leaf
    pale = cv2.bitwise_and(white_pixels, leaf_mask)

    # Desaturated green (early chlorosis)
    chlorotic_green = cv2.inRange(hsv, (35, 0, 120), (85, 60, 255))

    color_candidates = cv2.bitwise_or(yellows, pale)
    color_candidates = cv2.bitwise_or(color_candidates, chlorotic_green)

    # Cela empêche de sélectionner les parties saines même si elles sont brillantes
    lower_green = np.array([30, 70, 50])
    upper_green = np.array([80, 255, 200])
    healthy_green = cv2.inRange(hsv, lower_green, upper_green)

    color_candidates = cv2.bitwise_and(color_candidates, cv2.bitwise_not(healthy_green))

    # 2. Détection de Texture (Stippling)
    # On remonte le seuil à 25 (15 était trop bas et captait le bruit du capteur)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap).astype(np.uint8)
    _, texture_mask = cv2.threshold(texture, 25, 255, cv2.THRESH_BINARY)

    # 3. COMBINAISON "STRICTE" (ET logique)
    # Pour être une lésion d'acariens, la zone doit être:
    # (Décolorée) ET (Dans la feuille)
    # Note: On n'oblige pas la texture partout car certaines taches jaunes sont lisses,
    # mais on utilise la texture pour valider les zones ambiguës.

    # Approche hybride :
    # Zone A = Couleur Jaune/Blanc franche (Sûr à 100%)
    zone_A = cv2.bitwise_and(color_candidates, leaf_mask)

    # Zone B = Texture rugueuse (bruit possible)
    # On ne garde la texture QUE si elle chevauche une zone légèrement décolorée
    zone_B = cv2.bitwise_and(texture_mask, color_candidates)

    final_mask = cv2.bitwise_or(zone_A, zone_B)

    # Nettoyage : les piqûres d'acariens sont minuscules, on utilise un kernel très fin
    # pour ne pas effacer les détails, mais on vire les pixels isolés.
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return final_mask

# def septoria_mask(img, hsve, gray, mask):
#     # ==========================================================
#     # 1️⃣ Lighting normalization
#     # ==========================================================
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     L, A, B = cv2.split(lab)
#
#     clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
#     L = clahe.apply(L)
#
#     lab_norm = cv2.merge((L, A, B))
#     img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
#
#     blur = cv2.medianBlur(img_norm, 3)
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     H, S, V = cv2.split(hsv)
#
#     # ==========================================================
#     # 2️⃣ Robust green removal
#     # ==========================================================
#     lab2 = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
#     _, A2, _ = cv2.split(lab2)
#
#     # STRICTER Green Mask: Lowered upper bound from 110 to 115
#     # This reduces the chance of grabbing "muddy yellow" as green.
#     green_mask = cv2.inRange(A2, 0, 115)
#     non_green = cv2.bitwise_not(green_mask)
#
#     # ==========================================================
#     # 3️⃣ SPECIAL TARGETS (The Problem Solvers)
#     # ==========================================================
#
#     # Target 1: #938B72 (Grayish-Brown)
#     target_1 = cv2.inRange(hsv, (15, 30, 100), (30, 90, 180))
#
#     # Target 2: #464218 (Dark Olive)
#     target_2 = cv2.inRange(hsv, (20, 100, 40), (35, 255, 120))
#
#     # Target 3: #8D8F79 (Gray-Green)
#     target_3 = cv2.inRange(hsv, (25, 20, 100), (45, 75, 180))
#
#     # 🔴 NEW Target 4: "Dried Leaf / Khaki" (Based on your Hex Codes)
#     # Covers #AC9C32, #664B34, #796251
#     # These are Hue: 10-28, Saturation: Moderate-High, Value: Low-High
#     # We allow this to be VERY broad because we will FORCE it to stay.
#     target_4 = cv2.inRange(hsv, (10, 60, 50), (28, 200, 200))
#
#     # Group targets that are allowed to be "Greenish" or "Yellowish"
#     # These bypass the 'non_green' check
#     force_include = cv2.bitwise_or(target_2, target_3)
#     force_include = cv2.bitwise_or(force_include, target_4)  # Add Target 4 here
#
#     # Target 1 is brown enough to respect the green filter
#     target_1 = cv2.bitwise_and(target_1, non_green)
#
#     # Combine all special targets
#     special_targets = cv2.bitwise_or(target_1, force_include)
#     special_targets = cv2.bitwise_and(special_targets, mask)
#
#     # ==========================================================
#     # 4️⃣ Standard Detection (Brown/Necrosis)
#     # ==========================================================
#     # Standard brown
#     brown = cv2.inRange(hsv, (0, 40, 20), (30, 255, 220))
#
#     # Very dark necrosis
#     dark1 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
#
#     strong_candidate = cv2.bitwise_or(brown, dark1)
#     strong_candidate = cv2.bitwise_and(strong_candidate, non_green)
#     strong_candidate = cv2.bitwise_and(strong_candidate, mask)
#
#     # ==========================================================
#     # 5️⃣ Merge
#     # ==========================================================
#     # We skip "Light/Early" detection for this specific "dried" case
#     # as it introduces noise, but you can add it back if needed.
#
#     candidate = cv2.bitwise_or(strong_candidate, special_targets)
#
#     # Morphological Closing to fill holes in the large dried area
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Increased kernel size slightly
#     candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)
#
#     # ==========================================================
#     # 6️⃣ Final Contour Validation (The Logic Fix)
#     # ==========================================================
#     final = np.zeros_like(candidate)
#     contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     leaf_area = cv2.countNonZero(mask)
#
#     for c in contours:
#         area = cv2.contourArea(c)
#
#         if area < 5: continue
#
#         # If the spot is HUGE (massive necrosis), we keep it!
#         # Your previous code deleted it if area > 0.75
#         if area > 0.90 * leaf_area: continue
#
#         contour_mask = np.zeros_like(candidate)
#         cv2.drawContours(contour_mask, [c], -1, 255, -1)
#
#         # 1. Check Forced Inclusion
#         is_forced = cv2.countNonZero(cv2.bitwise_and(force_include, contour_mask)) > 0
#
#         # 2. Dynamic Green Ratio
#         # If the spot is large (> 5% of leaf), we are more lenient with the green ratio
#         # because large necrotic spots often have yellow/olive centers.
#         if area > 0.05 * leaf_area:
#             ratio_threshold = 0.50  # Allow 50% "green-ish" pixels
#         else:
#             ratio_threshold = 0.20  # Strict for small spots
#
#         if not is_forced:
#             green_pixels = cv2.bitwise_and(green_mask, contour_mask)
#             green_ratio = cv2.countNonZero(green_pixels) / (area + 1e-5)
#
#             if green_ratio > ratio_threshold:
#                 continue
#
#         cv2.drawContours(final, [c], -1, 255, -1)
#
#     return final


def septoria_mask(img, hsve, gray, mask):
    # ==========================================================
    # 1️⃣ Lighting normalization
    # ==========================================================
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Stronger contrast limit to separate mud-brown from green
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L = clahe.apply(L)

    lab_norm = cv2.merge((L, A, B))
    img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    # Gaussian blur is better for large dried patches than Median
    blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ==========================================================
    # 2️⃣ Robust Green Removal
    # ==========================================================
    lab2 = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB)
    _, A2, _ = cv2.split(lab2)

    # TIGHTER Green Mask.
    # Standard Green is A < 115. We lower it to 112 to stop
    # "Muddy Yellow" (A=113-115) from being counted as healthy.
    green_mask = cv2.inRange(A2, 0, 112)
    non_green = cv2.bitwise_not(green_mask)

    # ==========================================================
    # 3️⃣ SPECIAL TARGETS (The "Blind Spot" Fixes)
    # ==========================================================

    # --- Target A: The "Khaki/Yellow-Brown" (Fix for #AC9C32) ---
    # We extend Hue up to 35 (eating into Yellow-Green territory)
    # BUT we require higher saturation/value to confirm it's not shadow.
    target_khaki = cv2.inRange(hsv, (15, 50, 60), (35, 255, 255))

    # --- Target B: The "Desaturated Gray/Dead" (Fix for #796251) ---
    # This captures the top part of your leaf which is gray/dried.
    # Hue: 0-40 (Red to Yellow), Saturation: VERY LOW (10-80), Value: Medium-High
    target_gray = cv2.inRange(hsv, (0, 10, 60), (40, 90, 220))

    # --- Target C: Dark Olive / Brown ---
    target_olive = cv2.inRange(hsv, (20, 90, 40), (35, 255, 120))

    # Combine Force-Include Targets
    # These are allowed to exist even if they look "Greenish" to the algorithm
    force_include = cv2.bitwise_or(target_khaki, target_gray)
    force_include = cv2.bitwise_or(force_include, target_olive)

    # ==========================================================
    # 4️⃣ Standard Detection (Necrosis)
    # ==========================================================
    brown = cv2.inRange(hsv, (0, 40, 20), (30, 255, 220))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))

    strong_candidate = cv2.bitwise_or(brown, dark)
    strong_candidate = cv2.bitwise_and(strong_candidate, non_green)

    # ==========================================================
    # 5️⃣ Merge & Clean
    # ==========================================================
    # Add the forced targets (Khaki/Gray) back in
    candidate = cv2.bitwise_or(strong_candidate, force_include)
    candidate = cv2.bitwise_and(candidate, mask)  # Ensure we stay inside the leaf

    # Morphological Close: Connects the patchy dried spots into one solid mass
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel_close)

    # ==========================================================
    # 6️⃣ Final Filter
    # ==========================================================
    final = np.zeros_like(candidate)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_area = cv2.countNonZero(mask)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 5: continue

        # Create a mask just for this specific spot to test it
        contour_mask = np.zeros_like(candidate)
        cv2.drawContours(contour_mask, [c], -1, 255, -1)

        # 🛑 SAFETY CHECK: Is this a "Forced" color?
        # If the spot overlaps with our Khaki/Gray masks, we KEEP IT.
        # We do NOT check the green ratio. This is critical for large dried areas.
        overlap = cv2.bitwise_and(contour_mask, force_include)
        is_forced_type = cv2.countNonZero(overlap) > 10

        if is_forced_type:
            # It matches our special colors -> Keep it, no questions asked.
            cv2.drawContours(final, [c], -1, 255, -1)
        else:
            # Standard brown spot -> Check if it's actually just healthy green noise
            green_pixels = cv2.bitwise_and(green_mask, contour_mask)
            green_ratio = cv2.countNonZero(green_pixels) / (area + 1e-5)

            # If it's mostly green (and not Khaki/Gray), delete it
            if green_ratio < 0.35:
                cv2.drawContours(final, [c], -1, 255, -1)

    return final
def bacterial_spot_mask(leaf,hsv_image,gray, leaf_mask):
    """
    Detects Bacterial Spot by finding dark (necrotic) regions
    and yellow/brown regions on the leaf.
    """

    # --- 1. Detect the Black/Dark Brown Spots (Necrosis) ---
    # We ignore Hue (0-180) and focus on Low Value (Darkness)
    # Value < 80 usually captures black/dark brown spots well
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)

    # --- 2. Detect the Yellow/Brown Halos (Chlorosis) ---
    # Yellow is approx Hue 20-35. Brown/Red is approx Hue 0-20 and 160-180.
    # We combine them to catch the "sick" parts of the leaf that aren't black yet.
    lower_brown_yellow = np.array([10, 50, 50])
    upper_brown_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv_image, lower_brown_yellow, upper_brown_yellow)

    # Combine Black spots + Yellow regions
    disease_mask = cv2.bitwise_or(mask_black, mask_yellow)
    disease_mask = remove_green(hsv_image, disease_mask)
    # --- 3. Clean up ---
    # Only keep disease that is ON the leaf
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

    # Morphological opening to remove tiny noise (grainy pixels)
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)

    return disease_mask


def target_spot_mask(leaf,hsv, gray, leaf_mask):

    # 1️⃣ Brown detection (expanded but controlled)
    brown = cv2.inRange(hsv, (0, 25, 40), (40, 255, 220))

    # 2️⃣ Remove very dark shadow
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))
    brown = cv2.bitwise_and(brown, cv2.bitwise_not(dark))

    # 3️⃣ Add texture constraint
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(lap).astype(np.uint8)
    _, texture_mask = cv2.threshold(texture, 20, 255, cv2.THRESH_BINARY)

    candidate = cv2.bitwise_and(brown, texture_mask)

    # 4️⃣ Keep only inside leaf
    candidate = cv2.bitwise_and(candidate, leaf_mask)

    # 5️⃣ Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, 1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, 2)

    # 6️⃣ Contour filtering
    final = np.zeros_like(candidate)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaf_area = cv2.countNonZero(leaf_mask)

    for c in contours:
        area = cv2.contourArea(c)

        if area < max(5, 0.0005 * leaf_area) or area > 0.5 * leaf_area:
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * area / (peri * peri)

        if area < 0.01 * leaf_area and circularity < 0.2:
            continue

        cv2.drawContours(final, [c], -1, 255, -1)

    return final


def mosaic_virus_mask(hsv, gray, leaf_mask):
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green = cv2.bitwise_and(green, leaf_mask)

    # gray_nv = cv2.bitwise_and(gray, suppress_veins(gray))

    lbp = lbp_map(gray)
    mosaic = lbp_variance_mask(lbp, leaf_mask)

    combined = cv2.bitwise_and(mosaic, green)
    # combined = mosaic_sanity(combined, leaf_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, 1)

    return combined

def curl_virus_mask(img,hsv,gray, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lesion = cv2.inRange(hsv, (18,40,120), (45,255,255))
    return cv2.bitwise_and(lesion, mask)




# DISPATCHER
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

def enhance_contrast_clahe(hsv_image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the Value (V) channel of the HSV image to make dark spots pop out.
    """
    hsv_copy = hsv_image.copy()
    clahe_obj = cv2.createCLAHE(clipLimit=2, tileGridSize=(20, 20))
    hsv_copy[:, :, 2] = clahe_obj.apply(hsv_copy[:, :, 2])
    return hsv_copy

def get_lesion_mask(leaf, leaf_mask, disease_folder_name):
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)
    hsvE = enhance_contrast_clahe(hsv)
    gray = cv2.cvtColor(cv2.cvtColor(hsvE, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

    name = disease_folder_name.lower()

    for key, func in DISEASE_RULES.items():
        if key in name:
            return func(leaf,hsvE, gray, leaf_mask)

    return np.zeros_like(leaf_mask) # Fallback / Healthy


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

    # --- Create overlay ---
    overlay = image.copy()

    # Green leaf
    overlay[leaf_mask > 0] = (0.6 * overlay[leaf_mask > 0] + np.array([0, 255, 0]) * 0.4).astype(np.uint8)

    # Red lesions
    overlay[lesion_mask > 0] = (0.4 * overlay[lesion_mask > 0] + np.array([0, 0, 255]) * 0.6).astype(np.uint8)

    # Text
    cv2.putText(overlay, disease, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Side-by-side ---
    h1, w1, _ = image.shape
    h2, w2, _ = overlay.shape
    h = max(h1, h2)

    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image
    canvas[:h2, w1:w1 + w2] = overlay

    # Divider line
    cv2.line(canvas, (w1, 0), (w1, h), (255, 255, 255), 2)

    # Labels
    cv2.putText(canvas, "Original", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, "Detection", (w1 + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(debug_dir, img_name), canvas)


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
                leaf, leaf_mask = extract_leaf_mask(image)
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
    # """
    # Visual debugging tool for:
    # - GrabCut leaf segmentation
    # - Disease lesion detection (weak labels)
    #
    # Green  = Leaf
    # Red    = Detected lesions
    # """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image.")
        return

    # --- Step 1: Leaf Segmentation ---
    leaf, leaf_mask = extract_leaf_mask(image)
    # leaf2,leaf_mask2=grabcut_leaf_segmentation(image)
    if leaf is None:
        print("❌ GrabCut failed.")
        return

    # --- Step 2: Lesion Detection ---
    lesion_mask= get_lesion_mask(
        leaf=leaf,
        leaf_mask=leaf_mask,
        disease_folder_name=disease_folder_name
    )
    # lesion_mask2 = get_lesion_mask(
    #     leaf=leaf2,
    #     leaf_mask=leaf_mask2,
    #     disease_folder_name=disease_folder_name
    # )

    # --- Step 3: Visualization ---
    overlay = image.copy()
    # overlay2 = image.copy()

    # Leaf = Green
    overlay[leaf_mask > 0] = [0, 255, 0]
    # overlay2[leaf_mask2 > 0] = [0, 255, 0]

    # Lesions = Red (overwrites green where needed)
    overlay[lesion_mask > 0] = [0, 0, 255]
    # overlay2[lesion_mask2 > 0] = [0, 0, 255]

    # --- Display ---
    cv2.imshow("Original Image", image)
    cv2.imshow("Leaf Mask ", leaf_mask)
    cv2.imshow("Leaf  ", leaf)
    cv2.imshow("Lesion Mask (Weak Labels)", lesion_mask)
    cv2.imshow("Overlay (Green=Leaf, Red=Lesions)", overlay)
    # cv2.imshow("Leaf Mask 2(GrabCut)", leaf_mask2)
    # cv2.imshow("Leaf  2(GrabCut)", leaf2)
    # cv2.imshow("Lesion Mask2 (Weak Labels)", lesion_mask2)
    # cv2.imshow("Overlay2 (Green=Leaf, Red=Lesions)", overlay2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Uncomment this to run the full process
    build_dataset()

    # Uncomment this to test a single image
    # debug_segmentation(
    #     image_path="Datasets/dataset_yolo/images/train/Tomato_Spider_mites_Two_spotted_spider_mite/72deb702-3883-4023-8d27-936917947afb___Com.G_SpM_FL 8440.JPG",
    #     disease_folder_name="Tomato_Spider_mites_Two_spotted_spider_mite"
    # )
    #
    # BASE_DIR = r"Datasets/dataset_yolo/images/train/Tomato_Spider_mites_Two_spotted_spider_mite"
    #
    # for filename in os.listdir(BASE_DIR):
    #
    #     if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
    #         continue
    #     debug_segmentation(
    #         image_path = os.path.join(BASE_DIR, filename),
    #         disease_folder_name="Tomato_Spider_mites_Two_spotted_spider_mite"
    #     )
    # Pepper__bell___Bacterial_spot
    # Tomato_Bacterial_spot
    # Tomato_Septoria_leaf_spot
    # Tomato__Tomato_YellowLeaf__Curl_Virus
    # Tomato_Late_blight
    # Tomato__Target_Spot
    # Tomato_Leaf_Mold
    # Tomato__Tomato_mosaic_virus
    # Tomato_Early_blight
    # Potato___Early_blight
    # Potato___Late_blight
    # Tomato_Spider_mites_Two_spotted_spider_mite


