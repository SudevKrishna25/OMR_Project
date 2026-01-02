
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import random

# Seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32
IMG_SIZE = (32, 32)
EPOCHS = 20

def train_cnn():
    data_dir = "dataset/train_patches"
    
    # Empty: 3033, Filled: 1557 | total: 4590
    class_weights = {0: 0.76, 1: 1.47}
    
    print(f"Loading Dataset (Seed {SEED})...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )
    
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Robust Augmentation (Crucial for small data)
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1), # Handle registration jitter (+/- 3px)
    ])
    
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    
    print("Starting Training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1
    )
    
    model.save('omr_model.keras')
    print("Model saved to omr_model.keras")

if __name__ == "__main__":
    train_cnn()
