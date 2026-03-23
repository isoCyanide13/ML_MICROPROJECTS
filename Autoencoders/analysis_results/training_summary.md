═══════════════════════════════════════════════
        MODEL TRAINING PARAMETERS SUMMARY
═══════════════════════════════════════════════

Dataset         : MNIST (Keras)
Train Samples   : 60,000
Test Samples    : 10,000
Input Shape     : (28, 28, 1)

───────────────────────────────────────────────
SHARED PARAMETERS (Both Models)
───────────────────────────────────────────────
Epochs          : 200
Batch Size      : 32
Optimizer       : Adam
Learning Rate   : 0.0001
Latent Space Dim: 8
Conv Filters    : (32, 64, 64, 64)
Conv Kernels    : (3, 3, 3, 3)
Conv Strides    : (1, 2, 2, 1)

───────────────────────────────────────────────
AE SPECIFIC
───────────────────────────────────────────────
Loss Function   : MSE (Mean Squared Error)
Recon Weight    : N/A

───────────────────────────────────────────────
VAE SPECIFIC
───────────────────────────────────────────────
Loss Function   : Reconstruction Loss + KL Divergence
Recon Weight    : 100
Sampling        : Reparameterization Trick

───────────────────────────────────────────────
RESULTS SUMMARY
───────────────────────────────────────────────
                        AE          VAE
MSE                   : 0.0181      0.0190
PSNR (dB)             : 17.43       17.21
Silhouette Score      : 0.1729      0.1541
Inter-class Dist      : 63.9705     3.3015
Intra-class Variance  : 551.1104    1.2660
Latent Std            : 24.3108     1.1499
KL Divergence         : N/A         34.7583
KNN Accuracy          : 93.42%      93.75%
