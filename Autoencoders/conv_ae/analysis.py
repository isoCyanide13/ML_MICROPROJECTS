import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from autoencoder import Autoencoder
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
    plt.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_latent_space_comparison(latent_representations, sample_labels):
    """
    Plots 2D and 3D latent space side by side for comparison.
    For 2D latent space: uses label as the 3rd axis in the 3D plot.
    For 3D latent space: uses all 3 real dimensions.
    """
    fig = plt.figure(figsize=(18, 8))
    cmap = "rainbow"
    latent_dim = latent_representations.shape[1]

    # --- 2D Plot ---
    ax2d = fig.add_subplot(1, 2, 1)
    scatter_2d = ax2d.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        c=sample_labels,
        cmap=cmap,
        alpha=0.5,
        s=2
    )
    ax2d.set_title("2D Latent Space", fontsize=13)
    ax2d.set_xlabel("Latent Dimension 1")
    ax2d.set_ylabel("Latent Dimension 2")
    plt.colorbar(scatter_2d, ax=ax2d, label="Digit Class")

    # --- 3D Plot ---
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    if latent_dim >= 3:
        # Real 3rd latent dimension
        z_axis = latent_representations[:, 2]
        z_label = "Latent Dimension 3"
    else:
        # Use class label as z-axis to show class separation in 3D
        z_axis = sample_labels.astype(float)
        z_label = "Class Label"

    scatter_3d = ax3d.scatter(
        latent_representations[:, 0],
        latent_representations[:, 1],
        z_axis,
        c=sample_labels,
        cmap=cmap,
        alpha=0.5,
        s=2
    )
    ax3d.set_title("3D Latent Space", fontsize=13)
    ax3d.set_xlabel("Latent Dimension 1")
    ax3d.set_ylabel("Latent Dimension 2")
    ax3d.set_zlabel(z_label)
    plt.colorbar(scatter_3d, ax=ax3d, label="Digit Class", shrink=0.6)

    plt.suptitle("Latent Space Comparison — 2D vs 3D", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_latent_space_per_class(latent_representations, sample_labels):
    """
    Plots each digit class separately in both 2D and 3D
    so you can inspect per-class clustering clearly.
    """
    num_classes = len(np.unique(sample_labels))
    cmap = plt.get_cmap("rainbow", num_classes)
    latent_dim = latent_representations.shape[1]

    fig = plt.figure(figsize=(18, 8))

    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    for class_idx in range(num_classes):
        mask = sample_labels == class_idx
        color = [cmap(class_idx)]

        ax2d.scatter(
            latent_representations[mask, 0],
            latent_representations[mask, 1],
            c=color,
            alpha=0.5,
            s=2,
            label=str(class_idx)
        )

        if latent_dim >= 3:
            z_vals = latent_representations[mask, 2]
        else:
            z_vals = np.full(mask.sum(), class_idx, dtype=float)

        ax3d.scatter(
            latent_representations[mask, 0],
            latent_representations[mask, 1],
            z_vals,
            c=color,
            alpha=0.5,
            s=2,
            label=str(class_idx)
        )

    ax2d.set_title("2D — per class", fontsize=13)
    ax2d.set_xlabel("Latent Dimension 1")
    ax2d.set_ylabel("Latent Dimension 2")
    ax2d.legend(title="Digit", markerscale=4, loc="best", fontsize=8)

    ax3d.set_title("3D — per class", fontsize=13)
    ax3d.set_xlabel("Latent Dimension 1")
    ax3d.set_ylabel("Latent Dimension 2")
    ax3d.set_zlabel("Latent Dimension 3" if latent_dim >= 3 else "Class Label")
    ax3d.legend(title="Digit", markerscale=4, loc="best", fontsize=8)

    plt.suptitle("Per-class Latent Space — 2D vs 3D", fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    autoencoder = Autoencoder.load(r"model")
    x_train, y_train, x_test, y_test = load_mnist()

    # 1. Reconstruction comparison
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    # 2. Latent space — 2D vs 3D side by side
    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_latent_space_comparison(latent_representations, sample_labels)

    # 3. Per-class breakdown — 2D vs 3D side by side
    plot_latent_space_per_class(latent_representations, sample_labels)