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

# from ultralytics import YOLO
#
# def train():
#     model = YOLO("yolov8m-seg.pt")
#
#     model.train(
#         data="dataset.yaml",
#         epochs=100,
#         imgsz=256,
#         batch=24,            # auto batch size
#         optimizer="AdamW",
#         lr0=5e-4,
#         weight_decay=5e-4,
#         patience=20,
#         device="0",       # auto GPU / CPU
#         project="runs_segmentation",
#         name="plant_disease_seg",
#         exist_ok=True,
#         workers=8            # speed up loading
#     )
#
# if __name__ == "__main__":
#     train()
