from ultralytics import YOLO

def train():

    model = YOLO("yolov8m-seg.pt")

    model.train(
        data="dataset.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        patience=30,
        workers=8,
        exist_ok=True,
        project="runs_segmentation",
        name="plant_disease_seg_m",
    )

if __name__ == "__main__":
    train()
