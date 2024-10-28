# Import necessary libraries
import os
from pyts.image import GramianAngularField
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import cm
# Load data
train_files = glob.glob("Falling-Dataset/data/train/*.csv")
test_files = glob.glob("Falling-Dataset/data/test/*.csv")

train_data = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
test_data = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

# Filter normal and anomalous events
train_data_normal = train_data[train_data['anomaly'] == 0]
train_data_anomaly = train_data[train_data['anomaly'] == 1]

# Split normal events into training and validation sets
train_data, val_data = train_test_split(train_data_normal, test_size=0.1, random_state=42)
train_data = train_data.drop('anomaly', axis=1)
val_data = val_data.drop('anomaly', axis=1)
train_data_anomaly = train_data_anomaly.drop('anomaly', axis=1)

# Drop non-coordinate columns
columns_to_drop = ['010-000-024-033', '010-000-030-096', '020-000-032-221', '020-000-033-111']
train_data_coord = train_data.drop(columns=columns_to_drop)
val_data_coord = val_data.drop(columns=columns_to_drop)
train_data_anomaly_coord = train_data_anomaly.drop(columns=columns_to_drop)

# Normalize the data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_coord = pd.DataFrame(scaler.fit_transform(train_data_coord), columns=train_data_coord.columns)
val_data_coord = pd.DataFrame(scaler.transform(val_data_coord), columns=val_data_coord.columns)
train_data_anomaly_coord = pd.DataFrame(scaler.transform(train_data_anomaly_coord), columns=train_data_anomaly_coord.columns)

# Set up the GAF transformer and parameters
gaf = GramianAngularField(method='summation')
timeStep = 14  # Set the time window to 14 timesteps

# Function to create GAF images for each 14-timestep window, stack them, and apply a color filter
def create_gaf_images_with_color(data, timestep, path, image_prefix, max_images=None):
    os.makedirs(path, exist_ok=True)
    images = []

    # Loop over data in windows of 'timestep' size
    for i in range(0, len(data) - timestep + 1, timestep):
        # Initialize lists to hold GAF images for x, y, and z
        gaf_x_list = []
        gaf_y_list = []
        gaf_z_list = []

        # Loop to take 3 consecutive windows for each variable
        for j in range(3):
            window_x = data.iloc[i + j * timestep:i + (j + 1) * timestep, 0].to_numpy().reshape(1, -1)
            window_y = data.iloc[i + j * timestep:i + (j + 1) * timestep, 1].to_numpy().reshape(1, -1)
            window_z = data.iloc[i + j * timestep:i + (j + 1) * timestep, 2].to_numpy().reshape(1, -1)

            # Generate GAF images for each variable
            gaf_x = gaf.fit_transform(window_x)[0]
            gaf_y = gaf.fit_transform(window_y)[0]
            gaf_z = gaf.fit_transform(window_z)[0]

            # Append GAF images to their respective lists
            gaf_x_list.append(gaf_x)
            gaf_y_list.append(gaf_y)
            gaf_z_list.append(gaf_z)

        # Stack the GAF images vertically (3x12) for each variable
        stacked_gaf_x = np.concatenate(gaf_x_list, axis=0)  # Shape should be (36, 128)
        stacked_gaf_y = np.concatenate(gaf_y_list, axis=0)  # Shape should be (36, 128)
        stacked_gaf_z = np.concatenate(gaf_z_list, axis=0)  # Shape should be (36, 128)

        # Now stack the sets of GAF images horizontally (1x3 for each variable)
        combined_gaf_image = np.concatenate([stacked_gaf_x, stacked_gaf_y, stacked_gaf_z], axis=1)  # Shape (36, 384)
        combined_gaf_image = np.clip(combined_gaf_image, 0, 1)  # Ensure values are between 0 and 1

        # Resizing to 128x128
        combined_gaf_image_resized = np.array(Image.fromarray((combined_gaf_image * 255).astype(np.uint8)).resize((128, 128)))

        # Apply colormap to create an RGB image
        colored_image = cm.viridis(combined_gaf_image_resized / 255.0)[:, :, :3]  # RGB channels only
        img = Image.fromarray((colored_image * 255).astype(np.uint8))  # Convert to PIL Image

        # Save the image using PIL
        img.save(f"{path}/{image_prefix}_gaf_image_{i // timestep + 1}.png")
        images.append(img)

        # Check if we've reached the maximum number of images to save
        if max_images is not None and len(images) >= max_images:
            break  # Stop if we've reached the limit

    return np.array(images)

# Create directories if they do not exist
os.makedirs('GAF_IMG/TRAIN/images', exist_ok=True)
os.makedirs('GAF_IMG/TEST/images', exist_ok=True)
os.makedirs('GAF_IMG/ANOMALY/images', exist_ok=True)

# Create GAF images for each 14-timestep window in the training, validation, and anomaly data
train_gaf_images = create_gaf_images_with_color(train_data_coord, timeStep, path='GAF_IMG/TRAIN/images', image_prefix='train', max_images=500)
val_gaf_images = create_gaf_images_with_color(val_data_coord, timeStep, path='GAF_IMG/TEST/images', image_prefix='val', max_images=100)
anom_gaf_images = create_gaf_images_with_color(train_data_anomaly_coord, timeStep, path='GAF_IMG/ANOMALY/images', image_prefix='ANOM')
