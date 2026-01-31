import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# We train on the NEW dataset created by the previous script
DATASET_DIR = "PlantVillage_Severity"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # Healthy, Mild, Moderate, Severe


def train_professional_model():
    # 1. Data Augmentation Strategy (Addressing "Robustness" promise)
    # We artificially create more difficult examples during training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values
        rotation_range=20,  # Rotate slightly
        width_shift_range=0.2,  # Shift left/right
        height_shift_range=0.2,  # Shift up/down
        shear_range=0.2,  # Slant the image
        zoom_range=0.2,  # Zoom in
        horizontal_flip=True,  # Flip left-right
        fill_mode='nearest',  # Handle empty pixels
        validation_split=0.2  # Use 20% of data for validation
    )

    # 2. Data Loaders
    print("🚀 Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # 3. Model Architecture (Custom CNN)
    # Designed to be explainable and robust
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classifier Head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Prevents overfitting (Regularization)
        Dense(NUM_CLASSES, activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Professional Callbacks
    # Save the absolute best version of the model, not just the last one
    checkpoint = ModelCheckpoint('best_plant_model.keras',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)

    # Stop if not improving for 5 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Slow down learning rate if we get stuck
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # 5. Start Training
    print("\n🧠 Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 6. Save Final Results Graph
    plot_history(history)
    print("✅ Training Complete. Model saved as 'best_plant_model.keras'")


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('training_results.png')
    plt.show()


if __name__ == "__main__":
    train_professional_model()