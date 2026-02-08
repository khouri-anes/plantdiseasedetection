import os
import shutil
import random

SOURCE_DIR = "Datasets/PlantVillage"
TARGET_DIR = "Datasets/dataset_yolo"
SPLIT_RATIO = 0.8  # 80% train, 20% val

random.seed(42)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        out_dir = os.path.join(TARGET_DIR, "images", split, cls)
        os.makedirs(out_dir, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(out_dir, img)
            )
