import os
import shutil
import random

SOURCE_DIR = "Datasets/PlantVillage"
TARGET_DIR = "Datasets/dataset_yolo2"
TRAIN_RATIO = 0.7  # 70% train
VAL_RATIO = 0.15  # 15% validation
TEST_RATIO = 0.15  # 15% test

random.seed(42)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_idx = int(total_images * TRAIN_RATIO)
    val_idx = train_idx + int(total_images * VAL_RATIO)

    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:val_idx]
    test_imgs = images[val_idx:]

    # Copy images to respective folders
    for split, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        out_dir = os.path.join(TARGET_DIR, "images", split, cls)
        os.makedirs(out_dir, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(out_dir, img)
            )

    # Optional: Print statistics
    print(f"Class: {cls}")
    print(f"  Total images: {total_images}")
    print(f"  Train: {len(train_imgs)} ({len(train_imgs) / total_images:.1%})")
    print(f"  Val: {len(val_imgs)} ({len(val_imgs) / total_images:.1%})")
    print(f"  Test: {len(test_imgs)} ({len(test_imgs) / total_images:.1%})")
    print()