from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/segment/runs_segmentation/plant_disease_seg_m/weights/best.pt")

def segment_image(image_path):
    results = model(image_path)

    image = cv2.imread(image_path)
    masks = []

    for r in results:
        if r.masks is None:
            continue

        for mask in r.masks.data:
            mask = mask.cpu().numpy()
            mask = (mask * 255).astype("uint8")
            masks.append(mask)

    return image, masks


if __name__ == "__main__":
    img, lesion_masks = segment_image("Datasets/dataset_yolo/images/val/Tomato_Spider_mites_Two_spotted_spider_mite/72deb702-3883-4023-8d27-936917947afb___Com.G_SpM_FL 8440.JPG")
    print(f"{len(lesion_masks)} lesion(s) détectée(s)")
