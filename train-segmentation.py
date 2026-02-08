from ultralytics import YOLO

def train():
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data="dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=8,            # auto batch size
        optimizer="AdamW",
        lr0=5e-4,
        weight_decay=5e-4,
        patience=20,
        device="cpu",       # auto GPU / CPU
        project="runs_segmentation",
        name="plant_disease_seg",
        exist_ok=True,
        workers=8            # speed up loading
    )

if __name__ == "__main__":
    train()
