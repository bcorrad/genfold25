import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from gtda.diagrams import PersistenceEntropy
from time import time
from typing import List, Dict
import matplotlib.cm as cm
from utils import load_real_dataset
import argparse, os

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")


def persistent_entropy_by_dimension(X: np.ndarray = None,
                                    n_perm_values: List[int] = range(10, 510, 10),
                                    maxdim: int = 2,
                                    epsilon: float = 0.005,
                                    **kwargs) -> Dict[int, List[float]]:
    """
    Calculates persistent entropy separately for each homological dimension,
    with automatic detection of stabilization using entropy derivative.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
    - n_perm_values: List[int], different values of n_perm to test
    - maxdim: int, maximum homological dimension to compute
    - epsilon: float, threshold for stabilization detection (derivative)
    - kwargs: Additional keyword arguments, including 'suffix' for output file naming and destination_folder for saving results.

    Returns:
    - entropy_by_dim: Dict[int, List[float]], persistent entropies for each dimension
    """

    def generate_circle(n_points=300, noise=0.05):
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = np.cos(theta) + noise * np.random.randn(n_points)
        y = np.sin(theta) + noise * np.random.randn(n_points)
        return np.vstack((x, y)).T

    X = generate_circle() if X is None else X
    suffix = kwargs.get('suffix', '')
    destination_folder = kwargs.get('destination_folder', '.')

    entropy_by_dim = {dim: [] for dim in range(maxdim + 1)}
    computation_times = []

    for n in n_perm_values:
        print(f"Calculating for n_perm={n}...")
        start_time = time()
        result = ripser(X, maxdim=maxdim, n_perm=n)
        elapsed_time = time() - start_time
        computation_times.append(elapsed_time)

        dgms = result['dgms']

        for dim in range(maxdim + 1):
            dgm = dgms[dim]
            if dgm.size == 0 or np.isnan(dgm).any() or np.isinf(dgm).any():
                entropy_by_dim[dim].append(0)
                continue

            diagram = np.hstack((dgm, np.full((dgm.shape[0], 1), dim)))
            diagrams = diagram[np.newaxis, :, :]

            pe = PersistenceEntropy(normalize=True, n_jobs=-1, nan_fill_value=0.0)
            entropy = pe.fit_transform(diagrams)
            entropy_by_dim[dim].append(entropy[0][0])

    with open(os.path.join(destination_folder, f'persistent_entropy_{suffix}.txt'), "w") as f:
        for dim, values in entropy_by_dim.items():
            f.write(f"E{dim}:\n")
            f.write(", ".join(map(str, values)) + "\n\n")
        f.write("Computation times (s):\n")
        f.write(", ".join(map(str, computation_times)) + "\n")

    optimal_nperm_by_dim = {}
    for dim in entropy_by_dim:
        entropies = np.array(entropy_by_dim[dim])
        deriv = np.abs(np.gradient(entropies, n_perm_values))
        for i, d in enumerate(deriv):
            if d < epsilon:
                optimal_nperm_by_dim[dim] = n_perm_values[i]
                break
        else:
            optimal_nperm_by_dim[dim] = n_perm_values[-1]

    print("\nOptimal n_perm values by dimension (derivative < epsilon):")
    for dim, n_opt in optimal_nperm_by_dim.items():
        print(f"E{dim}: n_perm = {n_opt}")

    # Plot Persistent Entropy + Time
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = cm.get_cmap('tab10', maxdim + 1)
    for dim in entropy_by_dim:
        ax1.plot(n_perm_values, entropy_by_dim[dim], marker='o', label=f'E{dim}', color=colors(dim))
    ax1.set_xlabel('n_perm')
    ax1.set_ylabel('Persistent Entropy')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(n_perm_values, computation_times, color='gray', linestyle='--', label='Time')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend(loc='upper right')

    plt.title('Persistent Entropy by Homology Dimension and Computation Time vs n_perm')
    fig.tight_layout()
    plt.savefig(os.path.join(destination_folder, f'persistent_entropy_{suffix}.png'))
    plt.show()

    # Plot Entropy Derivatives
    fig, ax = plt.subplots(figsize=(10, 6))
    for dim in entropy_by_dim:
        entropies = np.array(entropy_by_dim[dim])
        deriv = np.abs(np.gradient(entropies, n_perm_values))
        ax.plot(n_perm_values, deriv, marker='o', label=f'dE{dim}/dn', color=colors(dim))
    ax.set_title('Numerical Derivative of Persistent Entropy vs n_perm')
    ax.set_xlabel('n_perm')
    ax.set_ylabel('Absolute Derivative of Persistent Entropy')
    ax.legend()
    ax.grid(True)
    # Save the threshold line
    ax.axhline(y=epsilon, color='red', linestyle='--', label=f'Epsilon = {epsilon}')
    ax.legend(loc='upper right')
    plt.title('Persistent Entropy Derivative by Homology Dimension')
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(destination_folder, f'persistent_entropy_derivative_{suffix}.png'))
    plt.show()
    # Save the derivative values
    with open(os.path.join(destination_folder, f'persistent_entropy_derivative_{suffix}.txt'), "w") as f:
        f.write("n_perm, " + ", ".join(map(str, n_perm_values)) + "\n")
        for dim in entropy_by_dim:
            entropies = np.array(entropy_by_dim[dim])
            deriv = np.abs(np.gradient(entropies, n_perm_values))
            f.write(f"dE{dim}/dn: " + ", ".join(map(str, deriv)) + "\n")

    return entropy_by_dim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute persistent entropy by homological dimension.")
    parser.add_argument('--dataset', type=str, default="MNIST", choices=["CIFAR10", "MNIST", "cifar10", "mnist"], help="Dataset to use")
    parser.add_argument('--n_perm_values', type=int, nargs='+', default=list(range(10, 510, 10)), help='List of n_perm values to test')
    parser.add_argument('--maxdim', type=int, default=2, help='Maximum homological dimension to compute')
    parser.add_argument('--epsilon', type=float, default=0.005, help='Threshold for stabilization detection')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output files')
    parser.add_argument("--class-ind", default=[3, 7], type=list, nargs='+', help="class index to use for the dataset, if None or empty, all classes are used")

    args = parser.parse_args()
    
    # Append the dataset name and classes to the suffix
    args.suffix += f"_{args.dataset.lower()}"
    args.suffix += f"_cls_{'_'.join(map(str, args.class_ind))}" if args.class_ind else ""
    args.suffix = ''.join(e for e in args.suffix if e.isalnum() or e in ('_', '-'))
    # Flatten class_ind if is a list of list
    if isinstance(args.class_ind, list) and all(isinstance(i, list) for i in args.class_ind):
        args.class_ind = [int(item) for sublist in args.class_ind for item in sublist]

    # Load the real dataset and calculate the persistence diagrams
    real_loader = load_real_dataset(dataset=args.dataset.upper(), cls=args.class_ind)
    
    # Loop over the real dataset
    real_dset = []
    for batch, _ in real_loader:
        # Concatenate the batch to the real dataset
        real_dset.append(batch.detach().cpu().numpy())
    real_dset = np.concatenate(real_dset, axis=0)
    # Flatten the images (N, C, H, W) -> (N, C*H*W)
    real_dset = real_dset.reshape(real_dset.shape[0], -1)
    folder = f'nperm_entropy_ablation_{args.dataset.lower()}_cls_{"-".join(map(str, args.class_ind))}_maxdim_{args.maxdim}_epsilon_{args.epsilon}'
    os.makedirs(folder, exist_ok=True)
    persistent_entropy_by_dimension(
        X=real_dset,
        n_perm_values=args.n_perm_values,
        maxdim=args.maxdim,
        epsilon=args.epsilon,
        suffix=args.suffix,
        destination_folder=folder
    )

# Call the function with your dataset
# nohup python /repo/corradini/GenFold/code/persistent_entropy_calculation.py --dataset MNIST --maxdim 2 --epsilon 0.005 --suffix mnist --class-ind 3 7 > persistent_entropy_mnist_37.log &