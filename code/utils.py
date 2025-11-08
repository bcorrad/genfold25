import os, random
import torch
import numpy as np

def set_gpu(gpu) -> bool:
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("Using GPU: {}".format(gpu))
        return f"cuda:{gpu}"
    else:
        print("Using CPU")
        return None


def set_seed(seed) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import List, Union
# Load real dataset based on args
def load_real_dataset(dataset: str=None,
                      data_root: Union[str, os.PathLike]="./data",
                      gan_batch_size: int=32, 
                      cls: list=[]) -> DataLoader:
    """
    Load a real dataset for GAN training.
    Args:
        dataset (str): Name of the dataset to load ('cifar10', 'cifar100', 'mnist').
        data_root (str): Root directory for the dataset.
        gan_batch_size (int): Batch size for the DataLoader.
        cls (list): List of classes to filter the dataset. If None, no filtering is applied.
    Returns:
        DataLoader: DataLoader for the specified dataset.
    """
    data_root += f"/{dataset.lower()}_{'_'.join(map(str, cls))}" if cls is not None and len(cls) > 0 else f"/{dataset.lower()}"
    if "cifar" in dataset.lower():
        import torchvision.datasets as ucifar
        dset = ucifar.CIFAR10(root=data_root,
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize((32, 32)),
                                  ]))
    elif dataset.lower() == "mnist":
        import torchvision.datasets as datasets
        dset = datasets.MNIST(data_root,
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Resize((32, 32)),
                                ]))

    if cls is not None and len(cls) > 0:
        cls_inds = [i for i, x in enumerate(dset.targets) if x in cls]
        dset = torch.utils.data.Subset(dset, cls_inds)

    return DataLoader(dset, batch_size=gan_batch_size, shuffle=True)

import os
import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt

def plot_persistent_entropy(entropy_file: str, derivative_file: str):
    def parse_entropy_file_robust(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        h0_values, h1_values, h2_values, computation_times = [], [], [], []
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('E0:'):
                current_section = 'E0'
                continue
            elif line.startswith('E1:'):
                current_section = 'E1'
                continue
            elif line.startswith('E2:'):
                current_section = 'E2'
                continue
            elif line.startswith('Computation times'):
                current_section = 'times'
                continue
            elif line == '' or line.startswith('#'):
                continue
            if current_section == 'E0' and line:
                h0_values.extend([float(x.strip()) for x in line.split(',') if x.strip()])
            elif current_section == 'E1' and line:
                h1_values.extend([float(x.strip()) for x in line.split(',') if x.strip()])
            elif current_section == 'E2' and line:
                h2_values.extend([float(x.strip()) for x in line.split(',') if x.strip()])
            elif current_section == 'times' and line:
                computation_times.extend([float(x.strip()) for x in line.split(',') if x.strip()])
        return h0_values, h1_values, h2_values, computation_times

    def parse_derivative_file_robust(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        n_permutations, dh0_dn, dh1_dn, dh2_dn = [], [], [], []
        for line in lines:
            line = line.strip()
            if line.startswith('n_perm'):
                parts = line.split(',')
                n_permutations = [int(x.strip()) for x in parts[1:] if x.strip()]
            elif line.startswith('dE0/dn:'):
                parts = line.split(',')
                dh0_dn = [float(x.strip()) for x in parts[1:] if x.strip()]
            elif line.startswith('dE1/dn:'):
                parts = line.split(',')
                dh1_dn = [float(x.strip()) for x in parts[1:] if x.strip()]
            elif line.startswith('dE2/dn:'):
                parts = line.split(',')
                dh2_dn = [float(x.strip()) for x in parts[1:] if x.strip()]
        return n_permutations, dh0_dn, dh1_dn, dh2_dn

    h0_values, h1_values, h2_values, computation_times = parse_entropy_file_robust(entropy_file)
    n_permutations, dh0_dn, dh1_dn, dh2_dn = parse_derivative_file_robust(derivative_file)

    # Estrai classi dal nome file
    dataset = "CIFAR10" if "cifar10" in entropy_file.lower() else "MNIST"
    classes = os.path.basename(entropy_file).split('cls_')[1].split('.txt')[0]
    classes = classes.split('_') if "_" in classes else [classes]

    # Per la scala log, sostituisci gli zeri con un valore piccolo
    eps = 1e-10
    h0_log = [v if v > 0 else eps for v in h0_values]
    h1_log = [v if v > 0 else eps for v in h1_values]
    h2_log = [v if v > 0 else eps for v in h2_values]
    dh0_log = [v if v > 0 else eps for v in dh0_dn]
    dh1_log = [v if v > 0 else eps for v in dh1_dn]
    dh2_log = [v if v > 0 else eps for v in dh2_dn]

    # Allinea le derivate (aggiungi uno zero iniziale)
    dh0_log = [eps] + dh0_log
    dh1_log = [eps] + dh1_log
    dh2_log = [eps] + dh2_log

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Entropie (linee continue)
    ax1.plot(n_permutations, h0_log, color='orange', linewidth=2, label=r'$E_0$')
    ax1.plot(n_permutations, h1_log, color='green', linewidth=2, label=r'$E_1$')
    ax1.plot(n_permutations, h2_log, color='blue', linewidth=2, label=r'$E_2$')

    # Derivate (linee tratteggiate)
    ax1.plot(n_permutations, dh0_log, color='orange', linestyle='--', linewidth=2, label=r'$\delta E_0/\delta n$')
    ax1.plot(n_permutations, dh1_log, color='green', linestyle='--', linewidth=2, label=r'$\delta E_1/\delta n$')
    ax1.plot(n_permutations, dh2_log, color='blue', linestyle='--', linewidth=2, label=r'$\delta E_2/\delta n$')
    # Add threshold for derivatives (0.005)
    # threshold = 0.005
    # ax1.axhline(y=threshold, color='red', linestyle=':', linewidth=2, label='Derivative threshold (0.005)')
    # Write labe in LaTeX
    ax1.set_xlabel(r'$n_\mathrm{perm}$', fontsize=15)
    ax1.set_ylabel(r'Persistent Entropy and its Derivatives', fontsize=15)
    ax1.set_xlim(0, max(n_permutations) + 10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=13, loc='lower right', ncol=2)

    # Tempi di calcolo (asse destro)
    ax2 = ax1.twinx()
    ax2.plot(n_permutations, computation_times, color='grey', linestyle='dotted', linewidth=2, label=r'$\Delta T$')
    ax2.set_ylabel('Computation Time (s)', fontsize=15, color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim(0, max(computation_times) + 5)
    ax2.legend(loc='upper right', fontsize=13)

    class_str = ', '.join(classes)
    if len(classes) > 1:
        class_str = f'Classes {class_str}'
    else:
        class_str = f'Class {class_str}'
    plt.title(
        fr'[{dataset} {class_str}] Optimal $n_\mathrm{{perm}}$ for Persistent Diagram Calculation',
        fontsize=16
    )
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(os.path.dirname(entropy_file), f'persistent_entropy_plot_{dataset}_{"_".join(classes)}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(f'persistent_entropy_plot_{dataset}_{"_".join(classes)}.png'), dpi=300, bbox_inches='tight')