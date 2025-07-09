# # SMAP & MSL Telemetry Anomaly Detection (normal data)
# # Foundation of Intelligent Systems Project
#
# **Author:** Daniela Chaves Acuña

# --- Section 1: Setup and Data Loading ---

# 1.1 Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # for listing files in directories

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # for visualization

# setting plotting style for better aesthetics

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# 1.2 Configuration

BASE_DATA_DIR = 'data/2018-05-19_15.00.10/'

TRAIN_DIR = BASE_DATA_DIR + 'train/'
TEST_DIR = BASE_DATA_DIR + 'test/'

RANDOM_STATE = 42

print(f"Starting SMAP & MSL Telemetry Anomaly Detection Project (Unsupervised).")
print(f"Loading data from: {TRAIN_DIR} and {TEST_DIR}")

# 1.3 Dynamically Load All Channel Datasets (.npy files)
# This approach will find all .npy files in the train and test directories
# and load them as individual channels.

def load_channels_from_dir(directory):
    """Loads all .npy files from a given directory and stacks them into a single numpy array."""
    channel_data_list = []
    channel_names = []
    
    # list all .npy files and sort them to ensure consistent column order
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in directory: {directory}")

    for filename in npy_files:
        file_path = os.path.join(directory, filename)
        try:
            data = np.load(file_path)
            # here we ensure data is 1D (time series for a single channel)
            if data.ndim != 1:
                data = data.flatten() # flatten to 1D if not already
            channel_data_list.append(data)
            channel_names.append(filename.replace('.npy', '')) # getting the channel name without .npy extension
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

    # stack the 1D channel arrays into a 2D array (time_points, num_channels)
    # then we have to ensure that all the channels have the same length before stacking
    min_len = min(len(arr) for arr in channel_data_list)
    channel_data_list = [arr[:min_len] for arr in channel_data_list]
    
    return np.stack(channel_data_list, axis=1), channel_names

try:
    X_train, train_channel_names = load_channels_from_dir(TRAIN_DIR)
    X_test, test_channel_names = load_channels_from_dir(TEST_DIR)
    
   
    if train_channel_names != test_channel_names:
        print("Warning: Train and Test directories contain different sets of channels or different order.")
        # lets assume they match or are sufficiently similar.
    
    CHANNEL_NAMES = train_channel_names # here we use train channel names as the reference
    
    print(f"\nSuccessfully loaded {X_train.shape[1]} channels for training and {X_test.shape[1]} channels for testing.")
    print(f"Example channel names: {CHANNEL_NAMES[:5]}...")
    
except FileNotFoundError as e:
    print(f"Error: Directory or .npy files not found. Please ensure BASE_DATA_DIR is correct: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Section 2: Dataset Overview and Initial Exploration ---

# 2.1 Display shapes of the loaded arrays

print(f"\n--- Dataset Shapes ---")
print(f"Shape of X_train (training telemetry data): {X_train.shape}") # (time_points, telemetry_channels)
print(f"Shape of X_test (testing telemetry data): {X_test.shape}")   # (time_points, telemetry_channels)

# Convert to Pandas DataFrames for easier handling and visualization prep

df_X_train = pd.DataFrame(X_train, columns=CHANNEL_NAMES)
df_X_test = pd.DataFrame(X_test, columns=CHANNEL_NAMES)


# 2.2 Display first few rows of the training DataFrame

print("\n--- Training Data Head (First 5 rows) ---")
print(df_X_train.head())

# 2.3 Get concise summary of the training DataFrame

print("\n--- Training Data Info ---")
df_X_train.info()

# 2.4 Generate descriptive statistics for training data

print("\n--- Training Data Description ---")
print(df_X_train.describe())

# 2.5 Check for missing values (usually not an issue with .npy files, but either way)

print("\n--- Missing Values Count in Training Data ---")
missing_values_train = df_X_train.isnull().sum()
print(missing_values_train[missing_values_train > 0])
if missing_values_train.sum() == 0:
    print("No missing values found in training data.")

print("\n--- IMPORTANT: No Ground Truth Labels for Direct Evaluation ---")
print("Based on the available files, true anomaly labels (y_test) are not provided for direct quantitative evaluation.")
print("This project will demonstrate unsupervised anomaly detection, relying on distance-based methods.")
print("Evaluation will be qualitative (visualizations) and based on the distribution of anomaly scores.")


# --- Section 3: Data Preprocessing (Feature Scaling) ---

# 3.1 Initialize StandardScaler

print("\n--- Applying Feature Scaling ---")
scaler = StandardScaler()

# 3.2 Fit the scaler on the training data (X_train) and transform both training and test data.

X_train_scaled = scaler.fit_transform(df_X_train)
X_test_scaled = scaler.transform(df_X_test)

print(f"Shape of scaled X_train: {X_train_scaled.shape}")
print(f"Shape of scaled X_test: {X_test_scaled.shape}")
print("Feature scaling completed successfully.")

# Display a sample of scaled data (first 5 rows, and first 5 columns)

print("\nFirst 5 rows of Scaled Training Data (sample):")
print(pd.DataFrame(X_train_scaled[:, :5], columns=CHANNEL_NAMES[:5]).head())


# --- Section 4: Determining Optimal K for K-Means (Elbow Method) ---

# 4.1 Calculate WCSS for a range of K values

print("\n--- Performing Elbow Method to find optimal K ---")
wcss = []
max_k = 10 # testing K from 1 to 10.

for i in range(1, max_k + 1):
    kmeans_elbow = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
    kmeans_elbow.fit(X_train_scaled) # fit on scaled training data
    wcss.append(kmeans_elbow.inertia_)

# 4.2 Plotting the Elbow Method results

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method for Optimal Number of Clusters (K)', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=14)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=14)
plt.xticks(range(1, max_k + 1))
plt.grid(True)
plt.show()
print("Elbow Method plot displayed. Analyze the 'elbow' to choose an optimal K.")


# --- Section 5: K-Means Clustering and Anomaly Detection ---

# 5.1 Choose Optimal K based on Elbow Method (visually pick from the plot)
# I first ran the script and with the help of the elbow method saw which K was the optimal one.
# in this case the line started to bend drastically in 4 so it is the chosen K.

OPTIMAL_K = 4
print(f"\n--- Applying K-Means Clustering with K = {OPTIMAL_K} ---")

# 5.2 Initialize and fit K-Means model on the scaled training data

# The model learns the clusters representing 'normal' behavior from the training data.
kmeans_model = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init='auto', random_state=RANDOM_STATE)
kmeans_model.fit(X_train_scaled)
print(f"K-Means model fitted on training data with {OPTIMAL_K} clusters.")

# 5.3 Predict cluster assignments for both training and test data

train_clusters = kmeans_model.predict(X_train_scaled)
test_clusters = kmeans_model.predict(X_test_scaled)

# 5.4 Calculate distance of each point to its assigned cluster centroid
# For anomaly detection, we measure how 'far' a point is from its learned normal cluster.
# The 'transform' method of KMeans gives distances to all centroids.
#  and then i take the minimum distance (distance to the assigned centroid).

distances_train = kmeans_model.transform(X_train_scaled)
min_distances_train = np.min(distances_train, axis=1) # distance to its assigned centroid

distances_test = kmeans_model.transform(X_test_scaled)
min_distances_test = np.min(distances_test, axis=1)   # distance to its assigned centroid

print("Distances to cluster centroids calculated for training and test data.")

# 5.5 Define an Anomaly Threshold
# A common approach is to use a statistical threshold based on the distances in the *training* data.
# For example, points with distances above a certain percentile or standard deviation from the mean distance.
# We'll use the 95th percentile of distances from the *training* data as our threshold.
# Points in the test set exceeding this threshold will be flagged as anomalies.

threshold_percentile = 95 # if you want to detect more (lower percentile) or fewer (higher percentile) anomalies
anomaly_threshold = np.percentile(min_distances_train, threshold_percentile)

print(f"\nAnomaly Detection Threshold (based on {threshold_percentile}th percentile of training distances): {anomaly_threshold:.4f}")

# 5.6 Detect anomalies in the test set

# Create a binary prediction array: 1 for anomaly, 0 for normal

y_pred_anomalies = (min_distances_test > anomaly_threshold).astype(int)

print(f"Detected {np.sum(y_pred_anomalies)} anomalies in the test set.")
print(f"This represents {np.sum(y_pred_anomalies) / len(y_pred_anomalies) * 100:.2f}% of the test data.")


# --- Section 6: Unsupervised Evaluation and Anomaly Score Visualization ---

# Since there are no ground truth labels (y_test), we focus on analyzing the distribution of anomaly scores and visualizing the detected outliers.

print("\n--- Unsupervised Anomaly Detection Evaluation ---")
print("Focusing on anomaly score distribution and visual inspection.")

# 6.1 Visualize the distribution of minimum distances (anomaly scores)

plt.figure(figsize=(10, 6))
sns.histplot(min_distances_test, bins=50, kde=True, color='skyblue', label='Min Distance to Centroid')
plt.axvline(anomaly_threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({threshold_percentile}th Percentile)')
plt.title('Distribution of Anomaly Scores (Min Distance to Cluster Centroid) in Test Data', fontsize=16)
plt.xlabel('Minimum Distance to Cluster Centroid', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
print("Anomaly score distribution plot displayed. Points beyond the red line are flagged as anomalies.")


# --- Section 7: Visualizations of Anomalies (using PCA) ---

print("\n--- Visualizing Detected Anomalies in 2D PCA Space ---")

# 7.1 Dimensionality Reduction for Visualization (using PCA from Week 2)

# We need to reduce the high-dimensional telemetry data to 2D for plotting.
# PCA transforms the data into a new set of orthogonal components that capture maximum variance.

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_test_pca = pca.fit_transform(X_test_scaled)
print(f"Test data reduced to 2 dimensions using PCA for visualization (shape: {X_test_pca.shape})")

# Create a DataFrame for easier plotting with predicted anomaly labels

df_plot = pd.DataFrame(X_test_pca, columns=['Principal Component 1', 'Principal Component 2'])
df_plot['Predicted_Anomaly'] = y_pred_anomalies
df_plot['Min_Distance'] = min_distances_test # keep distance for potential color mapping

# 7.2 Scatter plot of anomalies in 2D PCA space

plt.figure(figsize=(12, 8))

# Plot normal points in one color

plt.scatter(df_plot[df_plot['Predicted_Anomaly'] == 0]['Principal Component 1'],
            df_plot[df_plot['Predicted_Anomaly'] == 0]['Principal Component 2'],
            c='blue', label='Predicted Normal', s=20, alpha=0.6, edgecolors='w', linewidth=0.5)

# Plot predicted anomalies in another color (making them stand out)

plt.scatter(df_plot[df_plot['Predicted_Anomaly'] == 1]['Principal Component 1'],
            df_plot[df_plot['Predicted_Anomaly'] == 1]['Principal Component 2'],
            c='red', label='Predicted Anomaly', s=70, alpha=0.8, marker='X', edgecolors='k', linewidth=1.5)

plt.title('Detected Anomalies in 2D PCA Space (Unsupervised)', fontsize=18)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProject execution complete. Please review the generated plots and console outputs.")
