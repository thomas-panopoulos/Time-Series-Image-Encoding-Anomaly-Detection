import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Num GPUs Available: ", len(physical_devices))
    # Optionally set memory growth to avoid memory allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU available, using CPU.")

# Define constants
MODEL_PATH = 'saved_autoencoder_model_falling.h5'
SIZE1 = 128
SIZE2 = 128 #stacked
batch_size = 64

# Define image generators for training, validation, and anomaly detection
datagen = ImageDataGenerator(rescale=1./255)
'''datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=lambda x: x[:, :, :3]  # Ensure only 3 channels are used
)'''

train_generator = datagen.flow_from_directory(
    'GAF_IMG/TRAIN/',
    target_size=(SIZE1, SIZE2),
    batch_size=batch_size,
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    'GAF_IMG/TEST/',
    target_size=(SIZE1, SIZE2),
    batch_size=batch_size,
    class_mode='input'
)

anomaly_generator = datagen.flow_from_directory(
    'GAF_IMG/ANOMALY/',
    target_size=(SIZE1, SIZE2),
    batch_size=batch_size,
    class_mode='input'
)

# Function to build the autoencoder model
def build_autoencoder():
    model = Sequential()
    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE1, SIZE2, 3)))
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
    history = model.fit(
        train_generator,
        steps_per_epoch=500 // batch_size,
        epochs=1000,
        validation_data=validation_generator,
        validation_steps=75 // batch_size,
        shuffle=True
    )
    # Save the model after training
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Display model summary
model.summary()

# The rest of your code continues...


# Get a batch of images to test the model
data_batch = []
img_num = 0
while img_num <= train_generator.batch_index:
    data = next(train_generator)
    data_batch.append(data[0])
    img_num += 1

predicted = model.predict(data_batch[0])

# Display random images and their reconstructions
image_number = random.randint(0, predicted.shape[0] - 1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(data_batch[0][image_number])
plt.subplot(122)
plt.imshow(predicted[image_number])
plt.show()

# Evaluate the reconstruction error for normal and anomaly images
validation_error = model.evaluate(validation_generator)
anomaly_error = model.evaluate(anomaly_generator)

print("Recon. error for validation data (normal):", validation_error)
print("Recon. error for anomaly data:", anomaly_error)

# Build the encoder model for latent space extraction
encoder_model = Sequential()
encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE1, SIZE2, 3)))
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

import matplotlib.pyplot as plt

# Function to compute reconstruction error for a batch of images
def calculate_reconstruction_error(batch_images, model):
    errors = []
    for img in batch_images:
        img = img[np.newaxis, :, :, :]  # Add batch dimension
        reconstruction = model.predict(img)
        recon_error = model.evaluate(reconstruction, img, batch_size=1, verbose=0)[0]  # MSE error
        errors.append(recon_error)
    return errors

# Get a batch of normal (training) images
train_batch = next(train_generator)[0]

# Get a batch of validation (normal) images
val_batch = next(validation_generator)[0]

# Get a batch of anomalous images
anomaly_batch = next(anomaly_generator)[0]

# Calculate reconstruction errors for each dataset
train_errors = calculate_reconstruction_error(train_batch, model)
val_errors = calculate_reconstruction_error(val_batch, model)
anomaly_errors = calculate_reconstruction_error(anomaly_batch, model)

# Plotting the reconstruction errors
plt.figure(figsize=(10, 6))

# Histogram for training set errors
plt.hist(train_errors, bins=30, alpha=0.6, color='blue', label='Training Data')

# Histogram for validation set errors
plt.hist(val_errors, bins=30, alpha=0.6, color='green', label='Validation Data')

# Histogram for anomaly set errors
plt.hist(anomaly_errors, bins=30, alpha=0.6, color='red', label='Anomalous Data')

plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Reconstruction Error Distribution')
plt.legend(loc='upper right')

plt.show()


# Function to classify an image as normal or anomaly


def check_anomaly(img_path):
    try:
        # Open and convert the image to RGB to ensure 3 channels
        img = Image.open(img_path).resize((128, 128), Image.Resampling.LANCZOS)
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return False
    except Exception as e:
        print(f"Error opening image: {e}")
        return False

    img = np.array(img) / 255.0
    img = img[np.newaxis, :, :, :]  # Shape: (1, 128, 128, 3)

    encoded_img = encoder_model.predict(img)
    encoded_img = [np.reshape(img, (out_vector_shape,)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]
    reconstruction = model.predict(img)
    recon_error = model.evaluate(reconstruction, img, batch_size=1)[0]

    if (
        density < uninfected_values[0] - 100 * uninfected_values[1]
        or density > uninfected_values[0] + 100 * uninfected_values[1]
        or recon_error > uninfected_values[2] + 100 * uninfected_values[3]
    ):
        return True  # Image is an anomaly
    else:
        return False



'''def check_anomaly(img_path):
    try:
        # Open and resize the image with updated resampling
        img = Image.open(img_path).resize((128, 128), Image.Resampling.LANCZOS)
    except PermissionError as e:
        print(f"Permission denied: {e}")
        return False  # Return False for permission errors
    except Exception as e:
        print(f"Error opening image: {e}")
        return False  # Return False for other errors
    img = np.array(img)
    print(f"Image shape: {img.shape}")  # Add this line to debug

    img = np.array(img) / 255.0
    img = img[np.newaxis, :, :, :]

    encoded_img = encoder_model.predict(img)
    encoded_img = [np.reshape(img, (out_vector_shape,)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]
    reconstruction = model.predict(img)
    recon_error = model.evaluate(reconstruction, img, batch_size=1)[0]

    if density < uninfected_values[0] - 3 * uninfected_values[1] or density > uninfected_values[0] + 3 * uninfected_values[1] or recon_error > uninfected_values[2] + 3 * uninfected_values[3]:
        return True  # Image is an anomaly
    else:
        return False  # Image is NOT an anomaly
'''
# Test the anomaly detection on all anomalous images
anom_image_paths = glob.glob('GAF_IMG/ANOMALY/images/*')
normal_file_paths = glob.glob('GAF_IMG/TEST/images/*')
correctly_identified = 0
false_id = 0
for img_path in anom_image_paths:
    if check_anomaly(img_path):  # If it returns True, it means an anomaly was correctly identified
        correctly_identified += 1

# Calculate and print the percentage of correctly identified anomalies
total_anomalies = len(anom_image_paths)



for img_path in normal_file_paths:
    if check_anomaly(img_path):  # If it returns True, it means an anomaly was incorrectly identified
        false_id += 1

# Calculate and print the percentage of correctly identified anomalies
total_normal = len(normal_file_paths)


if total_normal > 0:
    accuracy_percentage_normal = (false_id / total_normal) * 100
    print(f"Falsely identified anomalies: {false_id}/{total_normal} ({accuracy_percentage_normal:.2f}%)")
else:
    print("No anomalous images found for testing.")
    
if total_anomalies > 0:
    accuracy_percentage = (correctly_identified / total_anomalies) * 100
    print(f"Correctly identified anomalies: {correctly_identified}/{total_anomalies} ({accuracy_percentage:.2f}%)")
else:
    print("No anomalous images found for testing.")