import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from conv_ae import Autoencoder
from conv_vae import VAE
from keras.datasets import mnist


# ─────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = x_test.astype("float32") / 255.0
    x_test  = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def select_images(images, labels, num_images=10):
    idx = np.random.choice(range(len(images)), num_images, replace=False)
    return images[idx], labels[idx]


# ─────────────────────────────────────────
# Plot 1: Reconstruction — two separate figures
# ─────────────────────────────────────────

def plot_reconstructed_images(images, ae_reconstructed, vae_reconstructed):
    """
    Two separate figures — one for AE, one for VAE.
    Each figure has 2 rows: original on top, reconstruction on bottom.
    AE reconstructions are typically sharper since there is no KL penalty.
    VAE reconstructions are slightly smoother but more consistently structured.
    """
    num_images = len(images)

    for recon, model_name in zip(
        [ae_reconstructed, vae_reconstructed],
        ["AE — Autoencoder", "VAE — Variational Autoencoder"]
    ):
        fig, axes = plt.subplots(2, num_images,
                                 figsize=(num_images * 1.8, 4))

        row_labels = ["Original", "Reconstructed"]
        rows = [images, recon]

        for row_idx, (label, row_imgs) in enumerate(zip(row_labels, rows)):
            for col_idx, img in enumerate(row_imgs):
                ax = axes[row_idx, col_idx]
                ax.imshow(img.squeeze(), cmap="gray_r")
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(label, fontsize=11, rotation=0,
                                  labelpad=50, va="center")

        plt.suptitle(f"Reconstruction — {model_name}", fontsize=13)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────
# Plot 2: 2D projection (Dim1 vs Dim2)
# ─────────────────────────────────────────

def plot_latent_2d(ae_latent, vae_latent, labels):
    """
    Projects the 8D latent space onto Dim1 vs Dim2.
    AE clusters will be scattered arbitrarily with no structure.
    VAE clusters will be compact and centered near the origin because
    the KL loss regularizes toward a standard normal distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = "rainbow"

    for ax, latent, title in zip(
        axes,
        [ae_latent, vae_latent],
        ["AE — Latent Space (Dim 1 vs Dim 2)",
         "VAE — Latent Space (Dim 1 vs Dim 2)"]
    ):
        sc = ax.scatter(latent[:, 0], latent[:, 1],
                        c=labels, cmap=cmap, alpha=0.5, s=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        plt.colorbar(sc, ax=ax, label="Digit Class")

    plt.suptitle("2D Projection (Dim 1 vs Dim 2) — AE vs VAE", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 3: Full 3D latent space (first 3 dims)
# ─────────────────────────────────────────

def plot_latent_3d(ae_latent, vae_latent, labels):
    """
    3D scatter using first 3 of 8 latent dimensions.
    AE: irregular blobs with no consistent origin or scale.
    VAE: clusters arranged in a structured gaussian ball around (0,0,0).
    """
    fig = plt.figure(figsize=(16, 7))
    cmap = "rainbow"

    for i, (latent, title) in enumerate(zip(
        [ae_latent, vae_latent],
        ["AE — 3D Latent Space (Dim 1,2,3)",
         "VAE — 3D Latent Space (Dim 1,2,3)"]
    )):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2],
                        c=labels, cmap=cmap, alpha=0.5, s=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        plt.colorbar(sc, ax=ax, label="Digit Class", shrink=0.5)

    plt.suptitle(
        "3D Latent Space (first 3 dims of 8) — AE vs VAE",
        fontsize=13
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 4: First 4 dimension projections
# Updated for latent_space_dim=8
# ─────────────────────────────────────────

def plot_all_projections(ae_latent, vae_latent, labels):
    """
    Shows 4 most informative 2D projections for both models.
    With 8 latent dims there are 28 possible projections — showing
    the first 4 pairs gives a good picture of the overall structure
    without being overwhelming.
    AE row on top, VAE row on bottom.
    """
    cmap = "rainbow"
    projections = [
        (0, 1, "Dim 1", "Dim 2"),
        (0, 2, "Dim 1", "Dim 3"),
        (1, 2, "Dim 2", "Dim 3"),
        (0, 3, "Dim 1", "Dim 4"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (xi, yi, xlabel, ylabel) in enumerate(projections):
        for row, (latent, model_name) in enumerate(zip(
            [ae_latent, vae_latent], ["AE", "VAE"]
        )):
            ax = axes[row, col]
            sc = ax.scatter(latent[:, xi], latent[:, yi],
                            c=labels, cmap=cmap, alpha=0.5, s=3)
            ax.set_title(f"{model_name} — {xlabel} vs {ylabel}", fontsize=11)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(sc, ax=ax, label="Digit")

    plt.suptitle(
        "Latent Dimension Projections — AE (top) vs VAE (bottom)",
        fontsize=13
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 5: Per-class cluster comparison (3D)
# ─────────────────────────────────────────

def plot_per_class_3d(ae_latent, vae_latent, labels):
    """
    Each digit class plotted with its own color using first 3 dims.
    Makes it easy to see which classes overlap or separate cleanly.
    VAE clusters should be tighter and better separated after retraining
    with reconstruction_loss_weight=100.
    """
    num_classes = 10
    cmap = plt.get_cmap("rainbow", num_classes)
    fig = plt.figure(figsize=(16, 7))

    for i, (latent, title) in enumerate(zip(
        [ae_latent, vae_latent],
        ["AE — Per-class 3D Clusters", "VAE — Per-class 3D Clusters"]
    )):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for cls in range(num_classes):
            mask = labels == cls
            ax.scatter(latent[mask, 0], latent[mask, 1], latent[mask, 2],
                       c=[cmap(cls)], alpha=0.5, s=3, label=str(cls))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend(title="Digit", markerscale=4, fontsize=7,
                  loc="upper left", framealpha=0.5)

    plt.suptitle("Per-class 3D Cluster Comparison — AE vs VAE", fontsize=13)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 6: Latent space density heatmap
# ─────────────────────────────────────────

def plot_density_comparison(ae_latent, vae_latent):
    """
    2D density heatmap using Dim1 vs Dim2.
    VAE with reconstruction_loss_weight=100 should now show a much
    tighter gaussian blob centered at (0,0) compared to before.
    AE will still show scattered hotspots with dead zones.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, latent, title in zip(
        axes,
        [ae_latent, vae_latent],
        ["AE — Latent Density (Dim 1 vs Dim 2)",
         "VAE — Latent Density (Dim 1 vs Dim 2)"]
    ):
        h = ax.hist2d(latent[:, 0], latent[:, 1], bins=60, cmap="hot")
        plt.colorbar(h[3], ax=ax, label="Point count")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")

    plt.suptitle(
        "Latent Density — AE (scattered) vs VAE (gaussian centered at origin)",
        fontsize=13
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 7: Reconstruction error per class
# ─────────────────────────────────────────

def plot_reconstruction_error(images, ae_reconstructed, vae_reconstructed, labels):
    """
    MSE per digit class for both models side by side.
    AE is optimized purely on MSE so it usually wins here.
    VAE optimizes reconstruction loss + KL divergence so it trades
    a little reconstruction sharpness for a structured latent space.
    With reconstruction_loss_weight=100 the gap may be slightly larger
    than before but the latent space quality will be much better.
    """
    num_classes = 10
    ae_errors  = []
    vae_errors = []

    for cls in range(num_classes):
        mask = labels == cls
        ae_errors.append(np.mean((images[mask] - ae_reconstructed[mask]) ** 2))
        vae_errors.append(np.mean((images[mask] - vae_reconstructed[mask]) ** 2))

    x     = np.arange(num_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, ae_errors,  width, label="AE",  color="#378ADD", alpha=0.8)
    ax.bar(x + width / 2, vae_errors, width, label="VAE", color="#E24B4A", alpha=0.8)

    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Reconstruction Error per Digit Class — AE vs VAE", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────

def compute_ae_metrics(images, reconstructed, latent, labels):
    """
    AE is trained purely on MSE so metrics reflect reconstruction quality
    and latent space structure with no regularization constraints.
    Works on any latent_space_dim size.
    """
    mse  = np.mean((images - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    subset = min(3000, len(latent))
    idx    = np.random.choice(len(latent), subset, replace=False)
    sil    = silhouette_score(latent[idx], labels[idx])

    num_classes = len(np.unique(labels))
    centroids   = np.array([
        latent[labels == cls].mean(axis=0) for cls in range(num_classes)
    ])
    pairwise   = cdist(centroids, centroids)
    inter_dist = pairwise[np.triu_indices(num_classes, k=1)].mean()
    intra_var  = np.mean([
        latent[labels == cls].var() for cls in range(num_classes)
    ])
    latent_std = latent.std()

    return {
        "MSE":                  round(float(mse), 6),
        "PSNR (dB)":            round(float(psnr), 2),
        "Silhouette Score":     round(float(sil), 4),
        "Inter-class Dist":     round(float(inter_dist), 4),
        "Intra-class Var":      round(float(intra_var), 4),
        "Latent Std":           round(float(latent_std), 4),
    }


def compute_vae_metrics(images, reconstructed, mu, log_variance, labels):
    """
    VAE is trained on reconstruction loss + KL divergence.
    With reconstruction_loss_weight=100, KL is now taken seriously
    and latent std should be much closer to 1.0 than before (was 6.0).
    Works on any latent_space_dim size.
    """
    recon_loss = np.mean((images - reconstructed) ** 2)
    psnr       = 10 * np.log10(1.0 / recon_loss) if recon_loss > 0 else float("inf")

    kl_loss = float(-0.5 * np.mean(
        np.sum(1 + log_variance
               - np.square(mu)
               - np.exp(log_variance), axis=1)
    ))
    total_loss = recon_loss * 100 + kl_loss  # matches new reconstruction_loss_weight

    subset = min(3000, len(mu))
    idx    = np.random.choice(len(mu), subset, replace=False)
    sil    = silhouette_score(mu[idx], labels[idx])

    num_classes = len(np.unique(labels))
    centroids   = np.array([
        mu[labels == cls].mean(axis=0) for cls in range(num_classes)
    ])
    pairwise   = cdist(centroids, centroids)
    inter_dist = pairwise[np.triu_indices(num_classes, k=1)].mean()
    intra_var  = np.mean([
        mu[labels == cls].var() for cls in range(num_classes)
    ])
    latent_std = mu.std()

    return {
        "Recon Loss":           round(float(recon_loss), 6),
        "KL Divergence":        round(float(kl_loss), 4),
        "Total VAE Loss":       round(float(total_loss), 4),
        "PSNR (dB)":            round(float(psnr), 2),
        "Silhouette Score":     round(float(sil), 4),
        "Inter-class Dist":     round(float(inter_dist), 4),
        "Intra-class Var":      round(float(intra_var), 4),
        "Latent Std":           round(float(latent_std), 4),
    }


# ─────────────────────────────────────────
# Plot 8: Performance metrics comparison
# ─────────────────────────────────────────

def plot_metrics_comparison(ae_metrics, vae_metrics):
    """
    Two side by side horizontal bar charts — one per model.
    AE shows MSE based metrics.
    VAE shows KL divergence + reconstruction loss + combined metrics.
    Shared comparison table below for metrics common to both models.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Performance Metrics — AE (MSE based) vs VAE (KL Divergence based)",
        fontsize=14
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[2, 1], hspace=0.4)

    # ── AE bar chart ──
    ax_ae  = fig.add_subplot(gs[0, 0])
    ae_keys = list(ae_metrics.keys())
    ae_vals = list(ae_metrics.values())
    bars = ax_ae.barh(ae_keys, ae_vals, color="#378ADD", alpha=0.8)
    for bar, val in zip(bars, ae_vals):
        ax_ae.text(bar.get_width() + max(ae_vals) * 0.01,
                   bar.get_y() + bar.get_height() / 2,
                   f"{val:.4f}", va="center", fontsize=9)
    ax_ae.set_title("AE Metrics", fontsize=12)
    ax_ae.set_xlabel("Value")
    ax_ae.grid(axis="x", alpha=0.3)
    ax_ae.invert_yaxis()

    # ── VAE bar chart ──
    ax_vae = fig.add_subplot(gs[0, 1])
    vae_keys = list(vae_metrics.keys())
    vae_vals = list(vae_metrics.values())
    bars = ax_vae.barh(vae_keys, vae_vals, color="#E24B4A", alpha=0.8)
    for bar, val in zip(bars, vae_vals):
        ax_vae.text(bar.get_width() + max(vae_vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
    ax_vae.set_title("VAE Metrics", fontsize=12)
    ax_vae.set_xlabel("Value")
    ax_vae.grid(axis="x", alpha=0.3)
    ax_vae.invert_yaxis()

    # ── Shared comparison table ──
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    shared = ["PSNR (dB)", "Silhouette Score", "Inter-class Dist",
              "Intra-class Var", "Latent Std"]

    higher_is_better = {
        "PSNR (dB)":        True,
        "Silhouette Score": True,
        "Inter-class Dist": True,
        "Intra-class Var":  False,
        "Latent Std":       None,   # closest to 1.0 wins
    }

    table_data = []
    for metric in shared:
        ae_val  = ae_metrics[metric]
        vae_val = vae_metrics[metric]
        hib     = higher_is_better[metric]

        if hib is True:
            winner = "VAE" if vae_val >= ae_val else "AE"
        elif hib is False:
            winner = "VAE" if vae_val <= ae_val else "AE"
        else:
            winner = "VAE" if abs(vae_val - 1.0) <= abs(ae_val - 1.0) else "AE"

        note = "higher better" if hib is True else \
               "lower better"  if hib is False else "closest to 1.0"

        table_data.append([metric, f"{ae_val:.4f}",
                           f"{vae_val:.4f}", winner, note])

    col_labels = ["Metric", "AE", "VAE", "Winner", "Note"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for row_idx, row in enumerate(table_data):
        winner = row[3]
        cell   = table[row_idx + 1, 3]
        cell.set_facecolor("#2ecc71" if winner == "VAE" else "#e74c3c")
        cell.set_text_props(color="white", fontweight="bold")

    ax_table.set_title(
        "Shared Metric Comparison — metrics that exist in both models",
        fontsize=11, pad=12
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Plot 9: Latent space accuracy (KNN)
# ─────────────────────────────────────────

def measure_latent_accuracy(ae_latent, vae_latent, labels):
    """
    Trains a KNN classifier on the latent representations of both models
    and measures how accurately digit classes can be predicted from the
    latent space alone.

    Higher accuracy means the latent space has learned a more meaningful
    and separable representation of the data.

    With latent_space_dim=8 and reconstruction_loss_weight=100, VAE should
    now comfortably beat AE — expected 88-93% vs AE's ~85%.
    """
    results = {}

    for latent, model_name in zip(
        [ae_latent, vae_latent], ["AE", "VAE"]
    ):
        x_tr, x_te, y_tr, y_te = train_test_split(
            latent, labels, test_size=0.2, random_state=42
        )
        knn      = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_tr, y_tr)
        accuracy = knn.score(x_te, y_te)
        results[model_name] = round(accuracy * 100, 2)
        print(f"  {model_name} Latent Space Accuracy: {accuracy * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    models     = list(results.keys())
    accuracies = list(results.values())
    colors     = ["#378ADD", "#E24B4A"]

    bars = ax.bar(models, accuracies, color=colors, alpha=0.85, width=0.4)
    for bar, val in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax.set_ylabel("KNN Classification Accuracy (%)")
    ax.set_title(
        "Latent Space Accuracy — KNN on latent representations\n"
        "Higher = more structured and separable latent space",
        fontsize=12
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Load models
    ae  = Autoencoder.load(r"D:\ML_MICROPROJECTS\Autoencoders\conv_ae\model")
    vae = VAE.load(r"D:\ML_MICROPROJECTS\Autoencoders\conv_vae\model")

    # Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # ── Plot 1: Reconstruction (two separate figures) ──
    num_display = 8
    sample_images, _ = select_images(x_test, y_test, num_display)
    ae_recon,  _          = ae.reconstruct(sample_images)
    vae_recon, _, _       = vae.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, ae_recon, vae_recon)

    # ── Plots 2-9: Latent space + metrics (large sample) ──
    num_latent = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_latent)

    ae_recon_full,  ae_latent              = ae.reconstruct(sample_images)
    vae_recon_full, vae_latent, vae_logvar = vae.reconstruct(sample_images)

    # Plot 2: 2D projection
    plot_latent_2d(ae_latent, vae_latent, sample_labels)

    # Plot 3: 3D latent space (first 3 dims)
    plot_latent_3d(ae_latent, vae_latent, sample_labels)

    # Plot 4: First 4 dimension projections
    plot_all_projections(ae_latent, vae_latent, sample_labels)

    # Plot 5: Per class 3D clusters
    plot_per_class_3d(ae_latent, vae_latent, sample_labels)

    # Plot 6: Density heatmap
    plot_density_comparison(ae_latent, vae_latent)

    # Plot 7: Reconstruction error per class
    plot_reconstruction_error(
        sample_images, ae_recon_full, vae_recon_full, sample_labels
    )

    # Plot 8: Performance metrics comparison
    print("\nComputing AE metrics...")
    ae_metrics = compute_ae_metrics(
        sample_images, ae_recon_full, ae_latent, sample_labels
    )

    print("Computing VAE metrics...")
    vae_metrics = compute_vae_metrics(
        sample_images, vae_recon_full, vae_latent, vae_logvar, sample_labels
    )

    print("\n── AE Metrics ──")
    for k, v in ae_metrics.items():
        print(f"  {k}: {v}")

    print("\n── VAE Metrics ──")
    for k, v in vae_metrics.items():
        print(f"  {k}: {v}")

    plot_metrics_comparison(ae_metrics, vae_metrics)

    # Plot 9: KNN latent space accuracy
    print("\nMeasuring latent space accuracy...")
    measure_latent_accuracy(ae_latent, vae_latent, sample_labels)
    