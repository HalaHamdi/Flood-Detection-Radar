import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FEATURE_NAMES = ["contrast", "homogeneity", "energy", "correlation"]
CHANNEL_NAMES = ["gray", "r", "g", "b"]

def apply_glcm(images, distance, positions):
    num_images = images.shape[0]
    num_channels = images.shape[3] + 1
    num_features = 4  # contrast, homogeneity, energy, correlation
    num_positions = len(positions)

    glcm_features = np.zeros((num_images, num_positions, num_channels, num_features))

    for i in tqdm(range(num_images)):
        for p in range(num_positions):
            gray_image = (rgb2gray(images[i])*255).astype(np.uint8)
            for c in range(num_channels):
                # Calculate GLCM
                if c == 0:
                    glcm = graycomatrix(gray_image, distances=[distance], angles=positions[p], levels=256, normed=True)
                else:
                    channel_image = (images[i, :, :, c-1]*255).astype(np.uint8)
                    glcm = graycomatrix(channel_image, distances=[distance], angles=positions[p], levels=256, normed=True)

                # Calculate GLCM features
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]

                glcm_features[i, p, c, 0] = contrast
                glcm_features[i, p, c, 1] = homogeneity
                glcm_features[i, p, c, 2] = energy
                glcm_features[i, p, c, 3] = correlation

    return glcm_features

def correlation_bet_positions(glcm_features):
    num_images, positions, num_channels, features = glcm_features.shape

    # Reshape the array to simplify the calculation
    glcm_features_reshaped = glcm_features.reshape(num_images, positions, num_channels * features)

    # Calculate the correlation matrix for each feature and channel
    correlation_matrices = []
    for f in range(features):
        for c in range(num_channels):
            channel_features = glcm_features_reshaped[:, :, c * features + f]
            correlation_matrix = np.corrcoef(channel_features, rowvar=False)
            correlation_matrices.append((f, c, correlation_matrix))

    # Print the correlation matrices
    for matrix in correlation_matrices:
        feature, channel, correlation = matrix
        print(f"Correlation for Feature {feature+1} and Channel {channel+1}:")
        print(correlation)
        print()


def plot_glcm_features(glcm_features, y, channel_index):
    print(f"Assessing Relation beteen each 2 features for the {CHANNEL_NAMES[channel_index]} channel")
    glcm_features_channel = glcm_features[:, channel_index, :]

    # Combine features and labels into a single DataFrame
    data = pd.DataFrame(glcm_features_channel, columns=[f"{CHANNEL_NAMES[channel_index]}_{FEATURE_NAMES[i]}" for i in range(glcm_features_channel.shape[1])])
    data['Label'] = y.copy()

    # Create a pairplot
    sns.set(style="ticks")
    plot = sns.pairplot(data, hue='Label', palette=["r", "b"])
    label_names = {0: 'Flooded', 1: 'Non-Flooded'}

    handles = plot._legend.legendHandles
    labels = label_names.values()
    plot._legend.remove()
    plot.fig.legend(handles=handles, labels=labels, title='Label')

    # Display the pairplot
    plt.show()

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Display the grid of graphs
    plt.show()