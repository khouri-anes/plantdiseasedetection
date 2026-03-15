import os
import shutil
import random
import cv2

SOURCE_DIR = "Datasets/PlantVillage"
TARGET_DIR = "Datasets/dataset_yolo2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET_PER_CLASS = 1500

random.seed(42)

# -------------------------
# AUGMENTATION FUNCTION
# -------------------------
def augment_image(image):

    aug = image.copy()
    choice = random.choice(["flip", "rotate", "scale"])

    if choice == "flip":
        aug = cv2.flip(aug, 1)

    elif choice == "rotate":
        angle = random.uniform(-20, 20)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        aug = cv2.warpAffine(aug, M, (w, h))

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

    images = [img for img in os.listdir(cls_path)
              if img.lower().endswith((".jpg",".png",".jpeg"))]

    random.shuffle(images)

    total_images = len(images)

    train_idx = int(total_images * TRAIN_RATIO)
    val_idx = train_idx + int(total_images * VAL_RATIO)

    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:val_idx]
    test_imgs = images[val_idx:]

    # -------------------------
    # CREATE DIRECTORIES
    # -------------------------
    train_dir = os.path.join(TARGET_DIR, "images", "train", cls)
    val_dir   = os.path.join(TARGET_DIR, "images", "val", cls)
    test_dir  = os.path.join(TARGET_DIR, "images", "test", cls)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # -------------------------
    # COPY ORIGINAL IMAGES
    # -------------------------
    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(test_dir, img))

    # -------------------------
    # AUGMENT TRAIN SET ONLY
    # -------------------------
    train_files = os.listdir(train_dir)

    if len(train_files) < TARGET_PER_CLASS:

        print(f"Augmenting {cls}: {len(train_files)} → {TARGET_PER_CLASS}")

        while len(train_files) < TARGET_PER_CLASS:

            img_name = random.choice(train_files)

            img_path = os.path.join(train_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            aug = augment_image(img)

            new_name = f"aug_{len(train_files)}_{img_name}"
            new_path = os.path.join(train_dir, new_name)

            cv2.imwrite(new_path, aug)

            train_files.append(new_name)

    # -------------------------
    # PRINT STATS
    # -------------------------
    print(f"\nClass: {cls}")
    print(f"Train: {len(os.listdir(train_dir))}")
    print(f"Val: {len(os.listdir(val_dir))}")
    print(f"Test: {len(os.listdir(test_dir))}")