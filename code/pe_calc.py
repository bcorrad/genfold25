import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from time import time
from typing import List, Dict, Union, Literal
import matplotlib.cm as cm
from tqdm import tqdm
import argparse, os
import warnings

warnings.filterwarnings("ignore")


def persistence_entropy(pd: np.ndarray, norm_mode: Union[Literal['persistence'], Literal['points']]='points') -> float:
    """
    Computes the persistence entropy of a given persistence diagram.
    Persistence entropy is a measure of the information content of a persistence diagram,
    which is commonly used in topological data analysis. It quantifies the distribution
    of persistence intervals in the diagram.
    Parameters:
    -----------
    pd : np.ndarray
        A 2D NumPy array representing the persistence diagram. Each row corresponds to a
        persistence interval, where the first column represents the birth time and the
        second column represents the death time. If the array has three columns, the third
        column is ignored.
    norm_mode : {'persistence', 'points'}, optional
        The normalization mode for the entropy calculation:
        - 'persistence': Normalizes by the total sum of the persistence intervals.
        - 'points': Normalizes by the number of persistence intervals.
        Default is 'points'.
    Returns:
    --------
    float
        The normalized persistence entropy of the persistence diagram. Returns 0.0 if
        there are no valid persistence intervals (e.g., all intervals have non-positive
        lengths).
    Raises:
    -------
    ValueError
        If `norm_mode` is not one of {'persistence', 'points'}.
    Notes:
    ------
    - Persistence intervals with non-finite or non-positive lengths are ignored.
    - The entropy is normalized using the specified `norm_mode` to ensure the result
        lies in a consistent range.
    Examples:
    ---------
    >>> import numpy as np
    >>> pd = np.array([[0.0, 1.0], [2.0, 5.0], [3.0, 4.0]])
    >>> persistence_entropy(pd, norm_mode='points')
    0.9182958340544896
    >>> persistence_entropy(pd, norm_mode='persistence')
    0.5435644431995964
    """
    if pd.shape[1] == 3:
        pd = pd[:, :2]

    lengths = pd[:, 1] - pd[:, 0]
    lengths = lengths[np.isfinite(lengths)]
    lengths = lengths[lengths > 0]

    if len(lengths) <= 1:
        return 0.0

    probs = lengths / lengths.sum()
    H = -np.sum(probs * np.log2(probs))
    if norm_mode == 'persistence':
        norm_factor = lengths.sum()
    elif norm_mode == 'points':
        norm_factor = len(lengths)
    else:
        raise ValueError("norm_mode must be 'persistence' or 'points'")
    return H / np.log2(norm_factor)


def persistent_entropy_by_dimension(
    X: np.ndarray,
    n_perm_values: List[int] = range(10, 510, 10),
    maxdim: int = 2,
    epsilon: float = 0.005,
    suffix: str = "",
    destination_folder: str = ".",
) -> Dict[int, List[float]]:
    """
    Calcola la Persistence Entropy (normalizzata, ∈ [0, 1]) per ciascuna
    dimensione omologica, usando la funzione custom.
    """

    entropy_by_dim = {dim: [] for dim in range(maxdim + 1)}
    computation_times = []

    for n in n_perm_values:
        print(f"Calcolo per n_perm={n}...")
        start_time = time()
        result = ripser(X, maxdim=maxdim, n_perm=n)
        computation_times.append(time() - start_time)

        dgms = result["dgms"]

        for dim, dgm in enumerate(dgms[: maxdim + 1]):
            if dgm.size == 0 or dgm.shape[0] <= 1:
                entropy_by_dim[dim].append(0.0)
                continue

            diagram = np.column_stack([dgm, np.full(len(dgm), dim)])
            val = persistence_entropy(diagram)
            entropy_by_dim[dim].append(val)

    # ---- Salvataggio su file ----
    os.makedirs(destination_folder, exist_ok=True)
    with open(os.path.join(destination_folder, f"persistent_entropy_{suffix}.txt"), "w") as f:
        for dim, values in entropy_by_dim.items():
            f.write(f"E{dim}:\n")
            f.write(", ".join(map(str, values)) + "\n\n")
        f.write("Computation times (s):\n")
        f.write(", ".join(map(str, computation_times)) + "\n")

    # ---- Rilevamento stabilizzazione ----
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

    print("\nValori ottimali di n_perm per dimensione (derivata < epsilon):")
    for dim, n_opt in optimal_nperm_by_dim.items():
        print(f"E{dim}: n_perm = {n_opt}")

    # ----- COLORI FISSI -----
    COL = {
        0: "#ffab26",   # E0 - arancione
        1: "#008001",   # E1 - verde
        2: "#0000ff",   # E2 - blu
        "time": "#7f7f7f",  # ΔT - grigio
    }

    fig, ax_ent = plt.subplots(figsize=(16, 10))

    # --------- ENTROPIE (asse sinistro, lineare) ----------
    for dim in sorted(entropy_by_dim):
        ax_ent.plot(
            n_perm_values, entropy_by_dim[dim],
            marker="o", linewidth=2, label=fr"$E_{dim}$", color=COL[dim]
        )
    ax_ent.set_xlabel(r"$n_{perm}$", fontsize=16, fontweight="bold")
    ax_ent.set_ylabel("Persistent Entropy", fontsize=16, fontweight="bold")
    ax_ent.tick_params(axis="both", labelsize=14)
    # Set x axis between min and max n_perm_values every 50
    ax_ent.set_xticks(range(0, max(n_perm_values) + 1, 50))
    # Add tick at 10 if not present
    if 10 not in ax_ent.get_xticks():
        ax_ent.set_xticks(list(ax_ent.get_xticks()) + [10])
        ax_ent.set_xticks(sorted(ax_ent.get_xticks()))
    ax_ent.set_xlim(min(n_perm_values), max(n_perm_values))
    ax_ent.grid(True, which="both", alpha=0.3)

    # --------- DERIVATE (asse sinistro OUTWARD, log) ----------
    ax_der = ax_ent.twinx()
    ax_der.spines["left"].set_position(("outward", 60))
    ax_der.yaxis.set_label_position("left")
    ax_der.yaxis.tick_left()
    for dim in sorted(entropy_by_dim):
        ent = np.array(entropy_by_dim[dim])
        der = np.abs(np.gradient(ent, n_perm_values))
        ax_der.plot(
            n_perm_values, der,
            linestyle="--", linewidth=2, label=fr"$\delta E_{dim}/\delta n$",
            color=COL[dim]
        )
    ax_der.set_ylabel(r"Derivatives $|dE/dn|$", fontsize=16, fontweight="bold")
    ax_der.set_yscale("log")
    ax_der.tick_params(axis="y", labelsize=14)
    # Scale axis between 10e-9 and 10e0
    ax_der.set_ylim(1e-9, 1)

    # --------- TEMPO (asse destro, lineare) ----------
    ax_time = ax_ent.twinx()
    ax_time.plot(
        n_perm_values, computation_times,
        color=COL["time"], linestyle=":", marker=".", linewidth=2, label=r"$\Delta T$"
    )
    ax_time.set_ylabel("Computation Time (s)", fontsize=16, fontweight="bold", color=COL["time"])
    ax_time.tick_params(axis="y", colors=COL["time"], labelsize=14)
    ax_time.spines['right'].set_color(COL["time"])

    # --------- LEGENDA sotto l'asse x (una riga) ----------
    lines, labels = [], []
    for ax in (ax_ent, ax_der, ax_time):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab

    fig.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),  # sotto la label x
        ncol=7,
        frameon=True,
        fontsize=14
    )
    plt.subplots_adjust(bottom=0.05)
    dataset = "MINST" if "mnist" in suffix.lower() else "CIFAR-10" if "cifar" in suffix.lower() else "CelebA"
    try:
        classes = suffix.split("cls_")[-1].replace("_", ", ")
        classes = classes.split("_")
        dataset += f" classes " + ", ".join(map(str, classes)) if len(classes) > 1 else f" class {classes[0]}"
    except Exception as e:
        print(f"An error occurred while processing the classes: {e}")
        dataset += " all classes"
    print(f"Dataset: {dataset}")
    print(f"Folder: {destination_folder}")

    # --------- TITOLO ----------
    plt.title(f"[{dataset}] "+"Optimal $n_{perm}$ for Persistent Diagram Calculation",
            fontsize=20, fontweight="bold")

    plt.savefig(os.path.join(destination_folder, f"persistent_entropy_combined_{suffix}.png"),
                bbox_inches="tight", dpi=300)
    plt.show()

    # ---- Salva derivate su file ----
    with open(os.path.join(destination_folder, f"persistent_entropy_derivative_{suffix}.txt"), "w") as f:
        f.write("n_perm, " + ", ".join(map(str, n_perm_values)) + "\n")
        for dim in entropy_by_dim:
            entropies = np.array(entropy_by_dim[dim])
            deriv = np.abs(np.gradient(entropies, n_perm_values))
            f.write(f"dE{dim}/dn: " + ", ".join(map(str, deriv)) + "\n")

    return entropy_by_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcolo della Persistent Entropy per dimensione omologica.")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["CIFAR10", "MNIST", "cifar10", "mnist", "CelebA", "celeba", "CELEBA"], help="Dataset da usare")
    parser.add_argument("--n_perm_values", type=int, nargs="+", default=list(range(10, 510, 10)), help="Valori di n_perm da testare")
    parser.add_argument("--maxdim", type=int, default=2, help="Massima dimensione omologica da calcolare")
    parser.add_argument("--epsilon", type=float, default=0.005, help="Soglia per rilevare stabilizzazione")
    parser.add_argument("--suffix", type=str, default="", help="Suffisso per i file di output")
    parser.add_argument("--class-ind", default=[3], type=int, nargs="+", help="Classi del dataset da usare")

    args = parser.parse_args()

    if args.class_ind == []:
        args.class_ind = None
        
    args.suffix += f"_{args.dataset.lower()}"
    args.suffix += f"_cls_{'_'.join(map(str, args.class_ind))}" if args.class_ind else ""
    args.suffix = "".join(e for e in args.suffix if e.isalnum() or e in ("_", "-"))

    if args.dataset.lower() == "celeba":
        from ID_ablation import load_real_dataset
        args.class_ind = args.class_ind if args.class_ind is not None and len(args.class_ind) > 0 else None
    else:
        from utils import load_real_dataset
        
    # Carica dataset reale
    real_loader = load_real_dataset(dataset=args.dataset.upper(), cls=args.class_ind, data_root="/repo/corradini/GenFold/data")

    real_dset = []
    for batch, _ in tqdm(real_loader):
        real_dset.append(batch.detach().cpu().numpy())
    real_dset = np.concatenate(real_dset, axis=0)
    real_dset = real_dset.reshape(real_dset.shape[0], -1)

    folder = f"miao_nperm_entropy_ablation_{args.dataset.lower()}_cls_{'-'.join(map(str, args.class_ind))}_maxdim_{args.maxdim}_epsilon_{args.epsilon}" if args.class_ind else f"nperm_entropy_ablation_{args.dataset.lower()}_all_maxdim_{args.maxdim}_epsilon_{args.epsilon}"
    os.makedirs(folder, exist_ok=True)

    persistent_entropy_by_dimension(
        X=real_dset,
        n_perm_values=args.n_perm_values,
        maxdim=args.maxdim,
        epsilon=args.epsilon,
        suffix=args.suffix,
        destination_folder=folder,
    )
