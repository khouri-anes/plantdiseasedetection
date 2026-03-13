import os
import shutil
import random
import cv2
import numpy as np

SOURCE_DIR = "Datasets/PlantVillage"
TARGET_DIR = "Datasets/dataset_yolo2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_PER_CLASS = 1200   # minimum images per class

random.seed(42)


# -------------------------
# AUGMENTATION FUNCTION
# -------------------------
def augment_image(image):

    aug = image.copy()

    choice = random.choice(["flip", "rotate", "brightness", "noise", "scale"])

    if choice == "flip":
        aug = cv2.flip(aug, 1)

    elif choice == "rotate":
        angle = random.uniform(-20, 20)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        aug = cv2.warpAffine(aug, M, (w, h))

    elif choice == "brightness":
        value = random.randint(-30, 30)
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + value, 0, 255)
        hsv = cv2.merge((h, s, v))
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif choice == "noise":
        noise = np.random.normal(0, 10, aug.shape).astype(np.uint8)
        aug = cv2.add(aug, noise)

    elif choice == "scale":
        scale = random.uniform(0.9, 1.1)
        h, w = aug.shape[:2]
        aug = cv2.resize(aug, None, fx=scale, fy=scale)
        aug = cv2.resize(aug, (w, h))

    return aug


# -------------------------
# MAIN LOOP
# -------------------------
for cls in os.listdir(SOURCE_DIR):

    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)

    # -------------------------
    # AUGMENT IF CLASS TOO SMALL
    # -------------------------
    if len(images) < TARGET_PER_CLASS:

        print(f"Augmenting class {cls} from {len(images)} → {TARGET_PER_CLASS}")

        while len(images) < TARGET_PER_CLASS:

            img_name = random.choice(images)
            img_path = os.path.join(cls_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            aug = augment_image(img)

            new_name = f"aug_{len(images)}_{img_name}"
            new_path = os.path.join(cls_path, new_name)

            cv2.imwrite(new_path, aug)

            images.append(new_name)

    # -------------------------
    # SPLIT DATASET
    # -------------------------
    random.shuffle(images)

    total_images = len(images)

    train_idx = int(total_images * TRAIN_RATIO)
    val_idx = train_idx + int(total_images * VAL_RATIO)

    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:val_idx]
    test_imgs = images[val_idx:]

    # -------------------------
    # COPY FILES
    # -------------------------
    for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:

        out_dir = os.path.join(TARGET_DIR, "images", split, cls)
        os.makedirs(out_dir, exist_ok=True)

        for img in split_imgs:

            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(out_dir, img)
            )

    # -------------------------
    # PRINT STATS
    # -------------------------
    print(f"\nClass: {cls}")
    print(f"Total images after augmentation: {total_images}")
    print(f"Train: {len(train_imgs)} ({len(train_imgs)/total_images:.1%})")
    print(f"Val: {len(val_imgs)} ({len(val_imgs)/total_images:.1%})")
    print(f"Test: {len(test_imgs)} ({len(test_imgs)/total_images:.1%})")