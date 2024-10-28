import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from sklearn.neighbors import KernelDensity

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Num GPUs Available: ", len(physical_devices))
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU available, using CPU.")

# Constants
MODEL_PATH = 'saved_autoencoder_model_falling2.h5'
SIZE1, SIZE2 = 128, 128  # Image dimensions
BATCH_SIZE = 64

# Image generators
datagen = ImageDataGenerator(rescale=1. / 255)
train_gen = datagen.flow_from_directory('GAF_IMG/TRAIN/', target_size=(SIZE1, SIZE2), batch_size=BATCH_SIZE, class_mode='input')
val_gen = datagen.flow_from_directory('GAF_IMG/TEST/', target_size=(SIZE1, SIZE2), batch_size=BATCH_SIZE, class_mode='input')
anomaly_gen = datagen.flow_from_directory('GAF_IMG/ANOMALY/', target_size=(SIZE1, SIZE2), batch_size=BATCH_SIZE, class_mode='input')

# Build autoencoder model with batch normalization
def build_autoencoder():
    model = Sequential()
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE1, SIZE2, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# Train or load model
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    model = build_autoencoder()
    model.fit(train_gen, steps_per_epoch=500 // BATCH_SIZE, epochs=100, validation_data=val_gen, validation_steps=75 // BATCH_SIZE, shuffle=True)
    model.save(MODEL_PATH)

# Extract the encoder part for latent space calculation
encoder = Sequential(model.layers[:6])  # Extract encoder layers only
encoder.summary()

# Calculate Kernel Density Estimation (KDE) for latent space distribution
encoded_images = encoder.predict(train_gen)
flat_encodings = encoded_images.reshape((encoded_images.shape[0], -1))
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(flat_encodings)

# Function to calculate density and reconstruction error
def calc_metrics(batch_images):
    densities, errors = [], []
    for img in batch_images:
        img = img[np.newaxis, :, :, :]
        encoded = encoder.predict(img).reshape(1, -1)
        density = kde.score_samples(encoded)[0]
        reconstructed = model.predict(img)
        error = model.evaluate(reconstructed, img, batch_size=1, verbose=0)[0]
        densities.append(density)
        errors.append(error)
    return np.mean(densities), np.std(densities), np.mean(errors), np.std(errors)

# Calculate metrics for training and anomaly data
train_batch = next(train_gen)[0]
anomaly_batch = next(anomaly_gen)[0]

train_metrics = calc_metrics(train_batch)
anomaly_metrics = calc_metrics(anomaly_batch)
print(f'Training Metrics: {train_metrics}')
print(f'Anomaly Metrics: {anomaly_metrics}')

# Function to classify image as normal or anomaly
def check_anomaly(img_path):
    try:
        img = Image.open(img_path).resize((128, 128), Image.Resampling.LANCZOS)
        img = np.array(img) / 255.0
        img = img[np.newaxis, :, :, :]

        encoded = encoder.predict(img).reshape(1, -1)
        density = kde.score_samples(encoded)[0]
        reconstructed = model.predict(img)
        error = model.evaluate(reconstructed, img, batch_size=1, verbose=0)[0]

        # Check if the image is an anomaly
        if (density < train_metrics[0] - 3 * train_metrics[1] or
                density > train_metrics[0] + 3 * train_metrics[1] or
                error > train_metrics[2] + 3 * train_metrics[3]):
            return True  # Anomaly detected
        else:
            return False  # Normal image

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

# Test anomaly detection on datasets
anom_paths = glob.glob('GAF_IMG/ANOMALY/images/*')
normal_paths = glob.glob('GAF_IMG/TEST/images/*')

correct, false_id = 0, 0

for path in anom_paths:
    if check_anomaly(path):
        correct += 1

for path in normal_paths:
    if check_anomaly(path):
        false_id += 1

# Calculate detection accuracy
total_anomalies = len(anom_paths)
total_normals = len(normal_paths)

if total_anomalies > 0:
    print(f"Correctly identified anomalies: {correct}/{total_anomalies} ({(correct / total_anomalies) * 100:.2f}%)")
if total_normals > 0:
    print(f"False positives: {false_id}/{total_normals} ({(false_id / total_normals) * 100:.2f}%)")

# Plot reconstruction errors for different datasets
def plot_reconstruction_errors(train_errors, val_errors, anomaly_errors):
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=30, alpha=0.6, color='blue', label='Training Data')
    plt.hist(val_errors, bins=30, alpha=0.6, color='green', label='Validation Data')
    plt.hist(anomaly_errors, bins=30, alpha=0.6, color='red', label='Anomalous Data')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

train_errors = calc_metrics(train_batch)[2]
val_errors = calc_metrics(next(val_gen)[0])[2]
anomaly_errors = calc_metrics(anomaly_batch)[2]

plot_reconstruction_errors(train_errors, val_errors, anomaly_errors)
