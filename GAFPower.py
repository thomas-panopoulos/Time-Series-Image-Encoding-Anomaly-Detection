import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the dataset
data = pd.read_csv('Power_Consumption/Tetuan City power consumption.csv')

# Handle missing values with forward fill
data = data.ffill()

# Normalize power consumption features
for zone in ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']:
    if zone in data.columns:
        data[zone] = (data[zone] - data[zone].mean()) / data[zone].std()

# Function to introduce anomalies with varying severity
def introduce_anomalies(data, anomaly_ratio=0.01, min_severity=2, max_severity=5):
    data_anomalous = data.copy()
    n_anomalies = int(len(data) * anomaly_ratio)

    for zone in ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']:
        if zone in data.columns:
            anomaly_indices = np.random.choice(len(data), n_anomalies, replace=False)
            
            # Generate random severities between min_severity and max_severity
            severities = np.random.uniform(min_severity, max_severity, size=n_anomalies)
            data_anomalous.loc[anomaly_indices, zone] *= np.random.choice([1, -1], size=n_anomalies) * severities

    return data_anomalous

# Create normal and anomalous datasets
data_anomalous = introduce_anomalies(data, anomaly_ratio=0.02, min_severity=1, max_severity=3)

# Set parameters for GAF
time_window = 128  # Updated to 128 timesteps
gaf = GramianAngularField(image_size=time_window)

# Create directories for Train, Validation, and Test GAF images
os.makedirs('GAF_Images/Normal/images', exist_ok=True)
os.makedirs('GAF_Images/Validation/images', exist_ok=True)
os.makedirs('GAF_Images/Anomalous/images', exist_ok=True)

# Function to create and save GAF images with a limit on the number
def create_gaf_images(data, label, max_images):
    image_count = 0  # Track the number of images created
    for zone in ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']:
        if zone in data.columns:
            zone_data = data[zone].values

            # Create sliding windows and generate GAF images
            for i in range(len(zone_data) - time_window + 1):
                if image_count >= max_images:
                    print(f"Reached the limit of {max_images} images for {label}.")
                    return

                segment = zone_data[i:i + time_window]
                gaf_image = gaf.fit_transform(segment.reshape(1, -1))

                # Apply 'jet' colormap and convert to RGB
                colored_image = cm.get_cmap('jet')(gaf_image[0])[:, :, :3]  # RGB channels only
                img = Image.fromarray((colored_image * 255).astype(np.uint8))

                # Save the image
                img.save(f'GAF_Images/{label}/images/{zone.replace(" ", "_")}_window_{i}.png')
                image_count += 1

    print(f"{image_count} {label} GAF images created.")

# Function to plot normal vs anomalous data on separate plots
def plot_data_comparison(data_normal, data_anomalous, zone):
    plt.figure(figsize=(14, 6))
    plt.plot(data_normal[zone].values, label='Normal Data', color='blue', alpha=0.7)
    plt.plot(data_anomalous[zone].values, label='Anomalous Data', color='red', alpha=0.7)
    plt.title(f'Comparison of {zone} - Normal vs Anomalous Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Power Consumption')
    plt.legend()
    plt.grid()
    plt.savefig(f'GAF_Images/{zone.replace(" ", "_")}_comparison.png')  # Save each plot as an image
    plt.show()  # Show the plot

# Generate and save GAF images
print("Generating GAF images for training (normal data)...")
#create_gaf_images(data, label='Normal', max_images=500)

print("Generating GAF images for validation (normal data)...")
#create_gaf_images(data, label='Validation', max_images=500)

print("Generating GAF images for test (anomalous data)...")
create_gaf_images(data_anomalous, label='Anomalous', max_images=100)

# Plot the comparison for each zone
for zone in ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']:
    if zone in data.columns:
        plot_data_comparison(data, data_anomalous, zone)

print("All GAF images for training, validation, and testing have been created and saved.")
