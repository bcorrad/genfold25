import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
from persim import wasserstein
import glob 
from typing import Union
from ID_ablation import load_real_dataset
import pandas as pd
G_FEATURES_FILENAME = "G_internal_representations"

# Suppress all warnings
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


def persistence(data_folder: Union[str, os.PathLike],
                save_folder: Union[str, os.PathLike],
                maxdim_H: int = 2,
                selected_epochs: list = None,
                **kwargs) -> None:
    """
    Compute and save persistence diagrams with two modes: layer-wise and epoch-wise.

    Parameters:
    - data_folder (str or os.PathLike): Folder containing the numpy files with internal representations.
    - save_folder (str or os.PathLike): Folder where the persistence diagrams will be saved.
    - maxdim_H (int): Maximum dimension for homology groups.
    - selected_epochs (list): List of epochs to compute persistence diagrams for.
    - kwargs: Additional keyword arguments, including 'n_perm' for Ripser. Default is 200.
    
    """
    n_perm = kwargs.get("n_perm", 100)  # Number of random permutations for Ripser
    save_folder = os.path.join(save_folder, "epoch-wise")
    os.makedirs(os.path.join(save_folder, "persistence_diagrams"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "persistence_diagrams_imgs"), exist_ok=True)
    rips = Rips(maxdim=maxdim_H, verbose=False, n_perm=n_perm)
    selected_epochs.append(-1)  # Add -1 to the list of selected epochs to compute the persistence diagram for the first epoch (input noise)
    perepoch_dgms = {}
    perepoch_wassdist_real = {}
    perepoch_wassdist_gen = {}
    dataset = data_folder.split("_dset_")[-1].split("_epochs")[0]   # Extract the dataset name between _dset_ and _epochs in the data_folder
    if "cls__" in data_folder:
        classes = []
    else:
        classes = data_folder.split("_cls_")[-1].split("_dt_")[0] 
        classes = [int(cls) for cls in classes.split("-")]
        
    # === Load the real dataset and calculate the persistence diagrams ===
    real_loader = load_real_dataset(dataset=dataset, cls=classes, data_root="/repo/corradini/GenFold/data/", max_num_samples=0.05)
    real_dset = []
    real_targets = []
    for batch, targets in tqdm(real_loader, desc="Processing real dataset batches"):
        real_dset.append(batch.detach().cpu().numpy())
        real_targets.append(targets.detach().cpu().numpy())
        if len(real_targets) * batch.shape[0] >= 10000:
            break
    real_dset = np.concatenate(real_dset, axis=0)
    real_dset = real_dset.reshape(real_dset.shape[0], -1)   # Flatten the images (N, C, H, W) -> (N, C*H*W)
    real_targets = np.concatenate(real_targets, axis=0)
    # real_dset = real_dset[real_targets == 1]
    
    if classes == []:
        # Pick 10% of the real dataset randomly if no classes are specified
        # np.random.seed(42)
        random_indices = np.random.choice(real_dset.shape[0], size=int(0.1 * real_dset.shape[0]), replace=False)
        real_dset = real_dset[random_indices]
        real_targets = real_targets[random_indices]
        
    real_dgms = rips.fit_transform(real_dset, distance_matrix=False)
    # Replace inf values with np.nan with a large number for better visualization
    for dim in range(len(real_dgms)):
        real_dgms[dim][real_dgms[dim] == np.inf] = np.nanmax(real_dgms[dim][real_dgms[dim] != np.inf]) * 1.1
    # Plot and save the persistence diagrams for the real dataset in a single plot using ripser plot_dgms
    rips.plot(real_dgms, show=False)
    # Save the plot
    plt.savefig(os.path.join(save_folder, "persistence_diagrams", "real_data_persistence_diagrams.png"))
    print(f"Saved plot: {os.path.join(save_folder, 'persistence_diagrams', 'real_data_persistence_diagrams.png')}")
    plt.close()
    # === Compute persistence diagrams for the selected epochs ===
    for e, epoch in enumerate(selected_epochs):
        epoch_data = []
        input_data = [] # To store the first layer features (i.e. noise) for the first epoch
        if epoch == -1:
            continue
        epoch_batches_features = np.load(os.path.join(data_folder, f"{G_FEATURES_FILENAME}_epoch_{epoch}.npy"), allow_pickle=True).item() # Features of all batches in the epoch
        first_layer_id, last_layer_id = list(epoch_batches_features[0].keys())[0], list(epoch_batches_features[0].keys())[-1]
        # === Collect features from all the batches in the epoch ===
        for batch_features in epoch_batches_features.values():
            if epoch == 0:
                input_data.append(batch_features[first_layer_id])
            epoch_data.append(batch_features[last_layer_id])
        if epoch == 0:
            perepoch_dgms[-1] = rips.fit_transform(np.concatenate(input_data, axis=0), distance_matrix=False)
        # === Compute the persistence diagrams for the current epoch ===
        epoch_data = np.concatenate(epoch_data, axis=0)
        epoch_data = epoch_data.reshape(epoch_data.shape[0], -1)
        if classes == []:
            # Pick 10% of the generated dataset randomly if no classes are specified
            random_indices = np.random.choice(epoch_data.shape[0], size=int(0.1 * epoch_data.shape[0]), replace=False)
            epoch_data = epoch_data[random_indices]
        perepoch_dgms[epoch] = rips.fit_transform(epoch_data, distance_matrix=False)
        # === Compute the Wasserstein distances vs. previous epoch and real data ===
        wasserstein_distance = [round(wasserstein(prev_dgm, curr_dgm), 3) for prev_dgm, curr_dgm in zip(perepoch_dgms[selected_epochs[e-1]], perepoch_dgms[epoch])]
        perepoch_wassdist_gen[epoch] = wasserstein_distance
        print(f"Wasserstein distance between epoch {selected_epochs[e-1]} and {epoch}: {wasserstein_distance}")
        real_wasserstein_distance = [round(wasserstein(real_dgm, curr_dgm), 3) for real_dgm, curr_dgm in zip(real_dgms, perepoch_dgms[epoch])]
        perepoch_wassdist_real[epoch] = real_wasserstein_distance
        print(f"Wasserstein distance between real data and epoch {epoch}: {real_wasserstein_distance}") 
    # === Save the persistence diagrams and Wasserstein distances ===
    if not os.path.exists(os.path.join(save_folder, "wasserstein_distances")):
        os.makedirs(os.path.join(save_folder, "wasserstein_distances"), exist_ok=True)
    pd.DataFrame(perepoch_wassdist_gen).to_csv(os.path.join(save_folder, "wasserstein_distances", "gen_wasserstein_distances.csv"), index=False)
    pd.DataFrame(perepoch_wassdist_real).to_csv(os.path.join(save_folder, "wasserstein_distances", "real_wasserstein_distances.csv"), index=False)
    # Remove -1 from the list of selected epochs after processing
    try:
        selected_epochs.remove(-1)
    except ValueError:
        pass
    # === Plot the Wasserstein distances W(H0), W(H1) and W(H2) for consecutive epochs ===
    for wassdist in [perepoch_wassdist_gen, perepoch_wassdist_real]:
        plt.figure(figsize=(10, 6))
        for dim in range(maxdim_H + 1):
            plt.plot(selected_epochs, [wassdist[epoch][dim] for epoch in selected_epochs], marker='o', label=f'W(H{dim})')
        plt.xticks(selected_epochs, rotation=45)
        title = "Wasserstein Distances vs. Consecutive Epochs" if wassdist == perepoch_wassdist_gen else "Wasserstein Distances vs. Real Data"
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Wasserstein Distance')
        plt.xticks(selected_epochs)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_folder, "persistence_diagrams_imgs", title.replace(" ", "_").lower() + f"_dim_{dim}.png"))
        print(f"Saved plot: {os.path.join(save_folder, 'persistence_diagrams_imgs', title.replace(' ', '_').lower() + f'_dim_{dim}.png')}")
        plt.close()


if __name__ == "__main__":
    path = "../../../homeRepo/corradini/experiments/"
    # path = "/repo/corradini/GenFold/code/experiments"
    for dir in os.listdir(path):
        if not "conv" in dir and not "celeba" in dir:
            print(f"Skipping {dir} as it does not match target.")
            continue
        print(f"Processing directory: {dir}")
        data_folder = os.path.join(path, dir, "results", "internal_representations")
        save_folder = os.path.join(path, dir, "PH_results")
        if not os.path.exists(data_folder):
            print(f"Data folder {data_folder} does not exist. Skipping...")
            continue
        if os.path.exists(save_folder):
            print(f"Save folder {save_folder} already exists. Skipping to avoid overwriting...")
            continue
        selected_epochs = list(range(0, 400, 10)) + [399]  # Example epochs from 0 to 399 with a step of 10
        try:
            persistence(data_folder, save_folder, selected_epochs=selected_epochs, n_perm=100)
        except Exception as e:
            print(f"Error processing {data_folder}: {e}")