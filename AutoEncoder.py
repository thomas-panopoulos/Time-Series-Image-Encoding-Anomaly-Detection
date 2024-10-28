import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

# Define constants
MODEL_PATH = 'saved_autoencoder_model.h5'
SIZE = 128
batch_size = 64

# Define image generators for training, validation, and anomaly detection
datagen = ImageDataGenerator(rescale=1./255, 
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    'GAF_Images/Normal/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    'GAF_Images/Validation/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
)

anomaly_generator = datagen.flow_from_directory(
    'GAF_Images/Anomalous/',
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='input'
)

# Function to build the autoencoder model
def build_autoencoder():
    model = Sequential()
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
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

# Check if the model is already saved, otherwise train a new one
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    model = build_autoencoder()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1000,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        shuffle=True,
        callbacks=[early_stopping]
    )
    # Save the model after training
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Display model summary
model.summary()

# Evaluate the reconstruction error for normal and anomaly images
validation_error = model.evaluate(validation_generator)
anomaly_error = model.evaluate(anomaly_generator)

print("Recon. error for validation data (normal):", validation_error)
print("Recon. error for anomaly data:", anomaly_error)

# Build the encoder model for latent space extraction
encoder_model = Sequential()
encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
encoder_model.layers[-1].set_weights(model.layers[0].get_weights())

encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
encoder_model.layers[-1].set_weights(model.layers[2].get_weights())

encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder_model.layers[-1].set_weights(model.layers[4].get_weights())
encoder_model.add(MaxPooling2D((2, 2), padding='same'))

encoder_model.summary()

# Calculate Kernel Density Estimation (KDE)
from sklearn.neighbors import KernelDensity

encoded_images = encoder_model.predict(train_generator)
encoder_output_shape = encoder_model.output_shape
out_vector_shape = np.prod(encoder_output_shape[1:])

encoded_images_vector = [np.reshape(img, (out_vector_shape,)) for img in encoded_images]
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)

# Function to calculate density and reconstruction error
def calc_density_and_recon_error(batch_images):
    density_list, recon_error_list = [], []
    for img in batch_images:
        img = img[np.newaxis, :, :, :]
        encoded_img = encoder_model.predict(img)
        encoded_img = [np.reshape(img, (out_vector_shape,)) for img in encoded_img]
        density = kde.score_samples(encoded_img)[0]
        reconstruction = model.predict(img)
        reconstruction_error = model.evaluate(reconstruction, img, batch_size=1)[0]
        density_list.append(density)
        recon_error_list.append(reconstruction_error)

    return (
        np.mean(density_list), np.std(density_list),
        np.mean(recon_error_list), np.std(recon_error_list)
    )

train_batch = next(train_generator)[0]
anomaly_batch = next(anomaly_generator)[0]

uninfected_values = calc_density_and_recon_error(train_batch)
anomaly_values = calc_density_and_recon_error(anomaly_batch)
print(f'normal: {uninfected_values}')
print(f'anomalous: {anomaly_values}')

# Function to classify an image as normal or anomaly
def check_anomaly(img_path):
    try:
        # Open and resize the image with updated resampling
        img = Image.open(img_path).resize((128, 128), Image.Resampling.LANCZOS)
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    img = np.array(img) / 255.0
    img = img[np.newaxis, :, :, :]

    encoded_img = encoder_model.predict(img)
    encoded_img = [np.reshape(img, (out_vector_shape,)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]
    reconstruction = model.predict(img)
    recon_error = model.evaluate(reconstruction, img, batch_size=1)[0]

    # Adjust the thresholds based on validation results
    density_threshold = 2500  # Adjust this threshold based on validation error analysis
    recon_error_threshold = 0.004  # Adjust based on analysis

    if density < density_threshold or recon_error > recon_error_threshold:
        print(f"The image {img_path} is an anomaly")
        return True  # Return True for anomaly
    else:
        print(f"The image {img_path} is NOT an anomaly")
        return False  # Return False for normal

# Test the anomaly detection on every anomalous image
para_file_paths = glob.glob('GAF_Images/Anomalous/images/*')
normal_file_paths = glob.glob('GAF_Images/Normal/images/*')
correctly_identified = 0
false_id = 0
for img_path in para_file_paths:
    if check_anomaly(img_path):  # If it returns True, it means an anomaly was correctly identified
        correctly_identified += 1

# Calculate and print the percentage of correctly identified anomalies
total_anomalies = len(para_file_paths)
if total_anomalies > 0:
    accuracy_percentage = (correctly_identified / total_anomalies) * 100
    print(f"Correctly identified anomalies: {correctly_identified}/{total_anomalies} ({accuracy_percentage:.2f}%)")
else:
    print("No anomalous images found for testing.")


for img_path in normal_file_paths:
    if check_anomaly(img_path):  # If it returns True, it means an anomaly was correctly identified
        false_id += 1

# Calculate and print the percentage of correctly identified anomalies
total_normal = len(normal_file_paths)
if total_normal > 0:
    accuracy_percentage_normal = (false_id / total_normal) * 100
    print(f"Falsely identified anomalies: {false_id}/{total_normal} ({accuracy_percentage_normal:.2f}%)")
else:
    print("No anomalous images found for testing.")