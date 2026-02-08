import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

def grabcut_leaf_segmentation(image):


    h, w = image.shape[:2]

    # --- Step 1: Initial mask ---
    # 0 = sure background
    # 1 = sure foreground
    # 2 = probable background
    # 3 = probable foreground
    mask = np.zeros((h, w), np.uint8)

    # Convert to HSV for initial guess
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Wide green + yellow-green range
    lower_leaf = np.array([20, 30, 30])
    upper_leaf = np.array([95, 255, 255])
    leaf_guess = cv2.inRange(hsv, lower_leaf, upper_leaf)

    mask[:] = cv2.GC_PR_BGD
    mask[leaf_guess > 0] = cv2.GC_PR_FGD

    # --- Step 2: Run GrabCut ---
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image,mask,
                None,
                bgdModel,
               fgdModel,
                5,
                cv2.GC_INIT_WITH_MASK
    )

    # --- Step 3: Extract leaf ---
    leaf_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # --- Step 4: Morphological cleanup ---
    kernel = np.ones((7, 7), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, 2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, 1)


    return leaf_mask



def segment_lesions(image, leaf_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Healthy green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green = cv2.inRange(hsv, lower_green, upper_green)

    # Lesion = non-green inside leaf
    lesion = cv2.bitwise_and(
        cv2.bitwise_not(green),
        leaf_mask
    )

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel, 1)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel, 2)

    return lesion



def segment_lesions(image, leaf_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Healthy green range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green = cv2.inRange(hsv, lower_green, upper_green)

    # Lesion = non-green inside leaf
    lesion = cv2.bitwise_and(
        cv2.bitwise_not(green),
        leaf_mask
    )

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel, 1)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel, 2)

    return lesion

def mask_to_yolo_polygons(mask, class_id):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = mask.shape
    labels = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue

        cnt = cnt.squeeze()
        if len(cnt.shape) != 2:
            continue

        polygon = []
        for x, y in cnt:
            polygon.append(x / w)
            polygon.append(y / h)

        labels.append(
            f"{class_id} " + " ".join(map(str, polygon))
        )

    return labels




CLASS_MAP = {
    "Pepper__bell___Bacterial_spot": 0,
    "Pepper__bell___healthy": 1,
    "Potato___Early_blight": 2,
    "Potato___Late_blight": 3,
    "Potato___healthy": 4,
    "Tomato_Bacterial_spot": 5,
    "Tomato_Early_blight": 6,
    "Tomato_Late_blight": 7,
    "Tomato_Leaf_Mold": 8,
    "Tomato_Septoria_leaf_spot": 9,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 10,
    "Tomato__Target_Spot": 11,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 12,
    "Tomato__Tomato_mosaic_virus": 13,
    "Tomato_healthy": 14
}

def build_yolo_segmentation_dataset(
    src_root="Datasets/dataset_yolo/images",
    dst_root="Datasets/PlantVillage_YOLO_SEG"
):
    for split in ["train", "val"]:
        img_out = os.path.join(dst_root, "images", split)
        lbl_out = os.path.join(dst_root, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        split_path = os.path.join(src_root, split)

        for cls in os.listdir(split_path):
            class_id = CLASS_MAP[cls]
            cls_path = os.path.join(split_path, cls)

            for img_name in tqdm(os.listdir(cls_path), desc=f"{split}/{cls}"):
                img_path = os.path.join(cls_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                leaf_mask = grabcut_leaf_segmentation(image)
                lesion_mask = segment_lesions(image, leaf_mask)

                labels = mask_to_yolo_polygons(lesion_mask, class_id)
                if not labels:
                    continue

                shutil.copy(
                    img_path,
                    os.path.join(img_out, img_name)
                )

                label_path = os.path.join(
                    lbl_out,
                    img_name.rsplit(".", 1)[0] + ".txt"
                )

                with open(label_path, "w") as f:
                    f.write("\n".join(labels))


def debug_grabcut(image_path):
    img = cv2.imread(image_path)
    leaf_mask = grabcut_leaf_segmentation(img)
    lesion_mask = segment_lesions(img, leaf_mask)

    overlay = img.copy()
    overlay[leaf_mask > 0] = [0, 255, 0]
    overlay[lesion_mask > 0] = [0, 0, 255]

    cv2.imshow("Original", img)
    cv2.imshow("Leaf Mask", leaf_mask)
    cv2.imshow("Lesions", lesion_mask)
    cv2.imshow("Overlay (Green=Leaf, Red=Lesion)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
     # build_yolo_segmentation_dataset()
     debug_grabcut(
         image_path="Datasets/PlantVillage_YOLO_SEG/images/train/2be7d064-fd61-49fe-95be-e07b37196bfc___YLCV_GCREC 2095.JPG"
     )