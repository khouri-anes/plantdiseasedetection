from ultralytics import YOLO

def train():


        model = YOLO("yolov8m-seg.pt")

        model.train(
            data="dataset.yaml",

            epochs=100,
            batch=32,
            imgsz=320,

            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,

            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,

            copy_paste=0.3,
            overlap_mask=True,
            label_smoothing=0.05,
            close_mosaic=10,

            patience=20,
            device=0,
            seed=42
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



    # model.train(
    #     data="dataset.yaml",
    #     project="runs_segmentation",
    #     name="plant_disease_best",
    #     exist_ok=True,
    #
    #     # ----- Epochs & Batch -----
    #     epochs=100,  # PlantVillage is moderate size; 150 epochs is safe
    #     batch=32,  # Adjust based on GPU memory (start with 16)
    #     imgsz=256,  # Standard YOLO size
    #     workers=8,
    #
    #     # ----- Optimizer & LR -----
    #     optimizer='AdamW',  # AdamW works well for segmentation
    #     lr0=0.001,  # Good starting point for AdamW
    #     lrf=0.01,  # Final LR = 0.001 * 0.01 = 1e-5
    #     momentum=0.937,  # Adam beta1 (or SGD momentum)
    #     weight_decay=0.0005,
    #     warmup_epochs=3,
    #     warmup_momentum=0.8,
    #     warmup_bias_lr=0.1,
    #     cos_lr=True,  # Cosine decay schedule
    #
    #     # ----- Regularization -----
    #     label_smoothing=0.1,  # Helps with ambiguous lesion boundaries
    #     dropout=0.0,  # Keep 0 unless you have massive overfitting
    #     overlap_mask=True,  # Allow masks to overlap (common in lesions)
    #
    #     # # ----- Data Augmentation (Tuned for leaves) -----
    #     # hsv_h=0.02,  # Slight hue shift (leaves are green)
    #     # hsv_s=0.7,  # Saturation variation (realistic)
    #     # hsv_v=0.4,  # Value variation
    #     degrees=10,  # Small rotations (leaves are not perfectly oriented)
    #     translate=0.1,  # Slight translation
    #     scale=0.5,  # Scale variation
    #     shear=2,  # Small shear (simulate perspective)
    #     perspective=0.0005,  # Minimal perspective
    #     flipud=0.1,  # Rare upside-down leaves
    #     fliplr=0.5,  # Horizontal flip is realistic
    #     # mosaic=1.0,  # Mosaic for first many epochs (good for context)
    #     # mixup=0.2,  # Mixup can improve generalization
    #     # copy_paste=0.3,  # Segment copy-paste aug (great for lesions)
    #     # auto_augment='randaugment',  # Auto augmentation policy
    #
    #     # ----- Other -----
    #     # close_mosaic=10,  # Disable mosaic in last 10 epochs to refine details
    #     val=True,
    #     save=True,
    #     save_period=10,
    #     patience=20,  # Early stopping if no improvement for 20 epochs
    #     device=0,
    #     seed=42,
    # )
    # model.tune(
    #     data="dataset.yaml",
    #     epochs=50,                # Each trial runs 50 epochs
    #     iterations=200,           # 200 different hyperparameter sets
    #     optimizer='AdamW',        # Fix optimizer, or omit to also tune it
    #     lr0=0.001,                # Optional: fix or let evolve
    #     plots=True,
    #     save=True,
    #     val=True,
    #     device=0,
    #     batch=32,                 # Use same batch as final training
    #     imgsz=256,
    #     workers=8,
    #     project="runs_tuning",
    #     name="tune_plant",
    # )