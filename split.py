import os
import shutil
import random
import cv2
import numpy as np

# ==========================
# CONFIG
# ==========================

import matplotlib.pyplot as plt
SOURCE_DIR = "Datasets/PlantVillage"
TARGET_DIR = "Datasets/dataset_yolo2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_IMAGES_PER_CLASS = 1500  # balance classes automatically

random.seed(42)

# ==========================
# AUGMENTATION FUNCTIONS
# ==========================

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


def flip(img):
    return cv2.flip(img, 1)


def brightness(img):
    return cv2.convertScaleAbs(img, alpha=1.2, beta=25)


def zoom(img):
    h, w = img.shape[:2]
    crop = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    return cv2.resize(crop, (w, h))


def augment_image(img):
    aug_type = random.choice(["rotate", "flip", "bright", "zoom"])

    if aug_type == "rotate":
        return rotate(img, random.choice([-20, -10, 10, 20]))
    if aug_type == "flip":
        return flip(img)
    if aug_type == "bright":
        return brightness(img)
    if aug_type == "zoom":
        return zoom(img)

    return img


# ==========================
# CREATE DIRECTORY STRUCTURE
# ==========================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(TARGET_DIR, "images", split), exist_ok=True)

# ==========================
# PROCESS DATASET
# ==========================

for cls in os.listdir(SOURCE_DIR):

    cls_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)

    train_idx = int(total * TRAIN_RATIO)
    val_idx = train_idx + int(total * VAL_RATIO)

    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:val_idx]
    test_imgs = images[val_idx:]

    print(f"\nProcessing class: {cls}")
    print("Original images:", total)

    # ==========================
    # COPY VAL + TEST
    # ==========================

    for split, split_imgs in [("val", val_imgs), ("test", test_imgs)]:

        out_dir = os.path.join(TARGET_DIR, "images", split, cls)
        os.makedirs(out_dir, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(out_dir, img)
            )

    # ==========================
    # TRAIN SET + AUGMENTATION
    # ==========================

    train_dir = os.path.join(TARGET_DIR, "images", "train", cls)
    os.makedirs(train_dir, exist_ok=True)

    count = 0

    for img_name in train_imgs:

        src = os.path.join(cls_path, img_name)
        dst = os.path.join(train_dir, img_name)

        shutil.copy(src, dst)
        count += 1

    # ==========================
    # BALANCE CLASSES
    # ==========================

    imgs_list = os.listdir(train_dir)

    while count < TARGET_IMAGES_PER_CLASS:

        img_name = random.choice(imgs_list)

        img_path = os.path.join(train_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        aug = augment_image(image)

        new_name = f"aug_{count}_{img_name}"

        cv2.imwrite(os.path.join(train_dir, new_name), aug)

        count += 1

    print("Train images after augmentation:", count)
    print("Validation images:", len(val_imgs))
    print("Test images:", len(test_imgs))

print("\nDataset preparation complete.")