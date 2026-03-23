import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from variational_autoencoder import VAE
from train import load_mnist


def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.suptitle("Original (top) vs Reconstructed (bottom)")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """
    Plots latent space in 3D if latent_space_dim >= 3,
    otherwise falls back to 2D.
    """
    if latent_representations.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(latent_representations[:, 0],
                             latent_representations[:, 1],
                             latent_representations[:, 2],
                             c=sample_labels,
                             cmap="rainbow",
                             alpha=0.5,
                             s=2)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_zlabel("Latent Dimension 3")
        plt.colorbar(scatter)
        plt.title("Latent Space 3D Visualization")
    else:
        # Fallback for latent_space_dim=2
        fig = plt.figure(figsize=(10, 10))
        scatter = plt.scatter(latent_representations[:, 0],
                              latent_representations[:, 1],
                              c=sample_labels,
                              cmap="rainbow",
                              alpha=0.5,
                              s=2)
        plt.colorbar(scatter)
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title("Latent Space 2D Visualization")
    plt.show()


if __name__ == "__main__":
    autoencoder = VAE.load(r"model")
    x_train, y_train, x_test, y_test = load_mnist()

    # Plot original vs reconstructed images
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    # Plot latent space
    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)