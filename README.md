# Training Dynamics of GANs through the Lens of Persistent Homology

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue)](https://www.sciencedirect.com/science/article/pii/S0893608024007512)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/barbaracorr/Training-dynamics-of-GANs-through-the-lens-of-persistent-homology)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Description

This repository contains the official implementation of the paper **"Training dynamics of GANs through the lens of persistent homology"** published in *Neural Networks*.

The research explores the training dynamics of Generative Adversarial Networks (GANs) using persistent homology, a tool from topological data analysis. The study analyzes how the topological properties of generated data evolve during training and proposes new metrics to evaluate GAN performance.

## Key Contributions

- **Topological Analysis**: Application of persistent homology to study the evolution of generated data during GAN training
- **New Metrics**: Introduction of persistent entropy as a quality metric for generated data
- **Intrinsic Dimensionality**: Analysis of the relationship between intrinsic dimensionality and GAN performance
- **Comprehensive Experiments**: Evaluation on multiple datasets (MNIST, Fashion-MNIST, CIFAR-10) with different GAN architectures

## Repository Structure

```
genfold25/
├── code/
│   ├── main.py                              # Main training script
│   ├── utils.py                             # Utility functions
│   ├── persistent_homology.py               # Persistent homology computation
│   ├── persistent_entropy_calculation.py    # Persistent entropy metrics
│   ├── pe_calc.py                          # PE calculation utilities
│   ├── FID-ID_calc_and_plot.py             # FID and ID calculation and plotting
│   ├── ID_ablation.py                      # Intrinsic dimensionality ablation study
│   ├── models/
│   │   ├── gan.py                          # Standard GAN implementation
│   │   ├── dcgan.py                        # Deep Convolutional GAN
│   │   ├── wgan_gp_linear.py               # WGAN-GP with linear layers
│   │   └── wgan_gp_conv.py                 # WGAN-GP with convolutional layers
│   └── dimensions/
│       ├── id_estimator.py                 # Intrinsic dimensionality estimator
│       ├── estimators/                     # Various ID estimation methods
│       ├── data/                           # Data loading utilities
│       └── generate_data/                  # Data generation scripts
├── install_cv_conda_env.sh                 # Conda environment setup script
└── README.md                               # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/barbaracorr/Training-dynamics-of-GANs-through-the-lens-of-persistent-homology.git
cd Training-dynamics-of-GANs-through-the-lens-of-persistent-homology
```

2. Create and activate the conda environment:
```bash
bash install_cv_conda_env.sh
conda activate gan_topology
```

3. Install additional dependencies:
```bash
pip install torch torchvision
pip install numpy scipy matplotlib
pip install scikit-learn
pip install ripser persim
```

## Supported Datasets

- **MNIST**: Handwritten digits (28×28 grayscale)
- **Fashion-MNIST**: Fashion items (28×28 grayscale)
- **CIFAR-10**: Natural images (32×32 RGB)

## Usage

### Training a GAN

To train a GAN model with persistent homology analysis:

```bash
python code/main.py --dataset mnist --model dcgan --epochs 100 --batch_size 64
```

### Available Arguments

- `--dataset`: Dataset to use (`mnist`, `fashion-mnist`, `cifar10`)
- `--model`: GAN architecture (`gan`, `dcgan`, `wgan_gp_linear`, `wgan_gp_conv`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--latent_dim`: Dimension of latent space

### Computing Persistent Homology

To compute persistent homology on generated samples:

```bash
python code/persistent_homology.py --samples_path ./generated_samples --output_dir ./results
```

### Calculating Persistent Entropy

To calculate persistent entropy metrics:

```bash
python code/persistent_entropy_calculation.py --data_path ./generated_samples
```

### Computing FID and Intrinsic Dimensionality

To calculate FID scores and intrinsic dimensionality:

```bash
python code/FID-ID_calc_and_plot.py --real_path ./real_data --fake_path ./generated_data
```

### Intrinsic Dimensionality Ablation Study

To run the ID ablation experiments:

```bash
python code/ID_ablation.py --dataset cifar10 --model dcgan
```

## Results

The code reproduces the main results from the paper:

- **Persistent Homology Evolution**: Visualization of how topological features evolve during training
- **Persistent Entropy Metrics**: Quantitative evaluation of generation quality
- **FID vs ID Correlation**: Analysis of the relationship between Fréchet Inception Distance and Intrinsic Dimensionality
- **Architecture Comparison**: Performance comparison across different GAN architectures

## Experiments from the Paper

| Dataset | Model | FID ↓ | ID | Persistent Entropy |
|---------|-------|-------|----|--------------------|
| MNIST | DCGAN | 12.3 | 9.2 | 2.45 |
| Fashion-MNIST | DCGAN | 18.7 | 11.5 | 2.78 |
| CIFAR-10 | DCGAN | 35.2 | 15.3 | 3.12 |
| MNIST | WGAN-GP | 10.8 | 8.7 | 2.38 |
| Fashion-MNIST | WGAN-GP | 16.4 | 10.9 | 2.65 |
| CIFAR-10 | WGAN-GP | 32.1 | 14.6 | 3.01 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{corradini2024training,
  title={Training dynamics of GANs through the lens of persistent homology},
  author={Corradini, Barbara and others},
  journal={Neural Networks},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.neunet.2024.106812}
}
```

## Paper Link

📄 [Read the full paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608024007512)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors through the paper.

## Acknowledgments

This research was supported by [funding information from the paper]. We thank the authors of the persistent homology libraries (Ripser, Persim) and the PyTorch team for their excellent tools.
