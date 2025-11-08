import argparse, os, time

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dimensions.id_estimator import run_estimator
from torchvision.datasets import ImageFolder 
from tqdm import tqdm

def set_seed(seed):
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def filter_by_class(dataset, cls):
    if cls is None or not isinstance(cls, (list, tuple)) or len(cls) == 0:
        return dataset
    try:
        targets = dataset.targets
    except AttributeError:
        return dataset
    cls_set = set(map(str, cls))
    indices = [i for i, t in enumerate(targets) if str(t) in cls_set]
    return torch.utils.data.Subset(dataset, indices)

def sample_subset(dataset, max_num_samples):
    if max_num_samples == -1 or max_num_samples >= len(dataset):
        return dataset
    inds = np.random.choice(len(dataset), size=max_num_samples, replace=False)
    return torch.utils.data.Subset(dataset, inds)


# COMMAND: nohup python -u /repo/corradini/GenFold/code/ID_ablation.py --dataset all --class-ind 3 --repeat 3 > all.log &
def load_real_dataset(dataset=None, train=True, max_num_samples=-1, data_root='./', batch_size=32, cls=None):
    import subprocess
    
    resize_all = transforms.Resize((32, 32))  # Adjust the size based on your model

    if "cifar" in dataset.lower():
        import torchvision.datasets as ucifar
        if dataset.lower() == "cifar100":
            dset_ = ucifar.CIFAR100
        else:
            dset_ = ucifar.CIFAR10

        dset = dset_(root=data_root,
                     train=train,
                     download=True,
                     transform=transforms.Compose([resize_all, transforms.ToTensor()]))
        
        logging.info(" " .join(map(str, ["Dataset size before filtering:", len(dset)])))

        if cls is not None and len(cls) > 0:
            cls_inds = [i for i, x in enumerate(dset.targets) if str(x) in cls or x in cls]
            dset = torch.utils.data.Subset(dset, cls_inds)
            
        logging.info(" " .join(map(str, ["Dataset size after filtering:", len(dset)])))
        if max_num_samples != -1:
            logging.info(" " .join(map(str, ["Sampling", max_num_samples, "samples"])))
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

        return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=torch.cuda.is_available())

    elif dataset.lower() == "mnist":
        import torchvision.datasets as datasets
        dset = datasets.MNIST(data_root,
                              download=True,
                              train=train,
                              transform=transforms.Compose([transforms.Grayscale(3), resize_all, transforms.ToTensor()]))

        if cls is not None and len(cls) > 0:
            logging.info(" " .join(map(str, ["Filtering class", cls])))
            cls_inds = [i for i, x in enumerate(dset.targets) if x in map(int, cls)]
            dset = torch.utils.data.Subset(dset, cls_inds)

        if max_num_samples != -1:
            logging.info(" " .join(map(str, ["Sampling", max_num_samples, "samples"])))
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

        return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=torch.cuda.is_available())

    elif "celeba" in dataset.lower():
            from torchvision.datasets.folder import default_loader

            celeba_root = os.path.join(data_root, "CelebA")
            img_dir = os.path.join(celeba_root, "img_align_celeba")
            partition_file = os.path.join(celeba_root, "Eval", "list_eval_partition.txt")

            # If CelebA is missing, run the bash downloader
            if not os.path.isdir(img_dir):
                logging.info(f"[INFO] CelebA not found (images or partition file missing).")
                # Download the CelebA dataset using the link: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
                celeba_zip = os.path.join(celeba_root, "celeba.zip")
                if not os.path.isfile(celeba_zip):
                    logging.info(f"[INFO] Downloading CelebA dataset...")
                    os.makedirs(celeba_root, exist_ok=True)
                    subprocess.run(["wget", "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip", "-O", celeba_zip])
                logging.info(f"[INFO] Extracting CelebA dataset...")
                subprocess.run(["unzip", "-o", celeba_zip, "-d", celeba_root])
                logging.info(f"[INFO] Extraction completed.")
                # if not os.path.isdir(img_dir) or not os.path.isfile(partition_file):
                #     raise RuntimeError(f"[ERROR] CelebA download or extraction failed. Please download and extract manually.")

            # Read partition file (0=train, 1=val, 2=test) with robust parsing
            partitions = {}
            with open(partition_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ls = s.lower()
                    if "<html" in ls or "</html>" in ls or "<head" in ls or "<body" in ls:
                        # Skip accidental HTML downloads
                        continue
                    parts = s.split()
                    if len(parts) < 2:
                        continue
                    fname, sid_str = parts[0], parts[-1]
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    try:
                        sid = int(sid_str)
                    except ValueError:
                        continue
                    partitions[fname] = sid

            if not partitions:
                raise RuntimeError(f"[ERROR] Could not parse entries from {partition_file}. It may be an HTML landing page.")

            # Choose split: train/test via 'train' flag; allow override with cls="train"/"val"/"test"
            split = None
            if isinstance(cls, str) and cls.lower() in {"train", "val", "test"}:
                split = cls.lower()
            else:
                split = "train" if train else "test"
            split_id = {"train": 0, "val": 1, "test": 2}[split]

            selected_files = [f for f, sid in partitions.items() if sid == split_id]
            if not selected_files:
                raise RuntimeError(f"[ERROR] No images matched the '{split}' split in CelebA.")

            # Use 64x64 for CelebA
            celeba_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])

            class CelebADataset(torch.utils.data.Dataset):
                def __init__(self, root, file_names, transform=None):
                    self.root = root
                    self.files = file_names
                    self.transform = transform

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    img_path = os.path.join(self.root, self.files[idx])
                    img = default_loader(img_path)
                    if self.transform:
                        img = self.transform(img)
                    return img, 0  # dummy label

            dset = CelebADataset(img_dir, selected_files, transform=celeba_transform)

            if cls is not None and isinstance(cls, (list, tuple)) and len(cls) > 0:
                logging.info("[WARNING] CelebA does not support class filtering. Ignoring cls parameter.")

            if max_num_samples != -1:
                logging.info(f"Sampling {max_num_samples} images from CelebA ({split} split)")
                all_inds = np.arange(len(dset))
                if type(max_num_samples) is float and 0 < max_num_samples < 1:
                    max_num_samples = int(len(dset) * max_num_samples)
                rand_inds = np.random.choice(all_inds, size=max_num_samples, replace=False)
                dset = torch.utils.data.Subset(dset, rand_inds)

            return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=torch.cuda.is_available())

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main(source_dir=None):
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--dataset", default="CelebA", type=str, choices=["CIFAR10", "MNIST", "ImageNet", "CelebA", "all"], help="Dataset to use")
    parser.add_argument("--anchor-samples", default=0, type=int, help="0 for using all samples from the training set")
    parser.add_argument("--anchor-ratio", default=0, type=float, help="0 for using all samples from the training set")
    parser.add_argument("--max_num_samples", default=-1, type=int, help="Maximum number of samples to process. Useful for evaluating convergence.")
    parser.add_argument("--n_cls", default=None, type=int, help="A redundant flag for specifying number of classes.")
    parser.add_argument("--separate-classes", default=False, action="store_true")
    parser.add_argument("--class-ind", default=[3], type=list, help="class index to use for the dataset, if None or empty, all classes are used")
    parser.add_argument("--splits", default=range(1000, 15000, 1000), type=list, help="Percentage of real images to use for intrinsic dimension calculation")
    parser.add_argument("--repeat", default=3, type=int, help="Number of times to repeat the experiment for averaging")
   
    # GAN Training args
    parser.add_argument("--n_epochs", default=400, type=int)
    parser.add_argument("--gan_batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=(32,32), type=int)
    parser.add_argument("--n_images", default=100000, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--bsize", default=1024, type=int, help="batch size for previous images")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--resize-size", default=(32, 32), type=tuple, help="resize size for the images")

    # Estimator-specific args
    ## MLE args
    parser.add_argument("--estimator", default="mle", type=str, choices=["mle", "geomle","twonn", "shortest-path"])
    parser.add_argument("--n-workers", default=1, type=int)
    parser.add_argument("--k1", default=10, type=int)
    parser.add_argument("--k2", default=55, type=int)
    parser.add_argument("--nb-iter", default=100, type=int)
    parser.add_argument("--eval-every-k", default=False, action="store_true", help="Whether to evaluate every k<=k1")
    parser.add_argument("--average-inverse", default=False, action="store_true", help="Whether to take the average of the inverse from each bootstrap run ")
    parser.add_argument("--single-k", default=True, action="store_true", help="Whether to estimate the dimension with a single k, if True reproduces the results of the paper")
    ## GeoMLE args
    parser.add_argument("--nb-iter1", default=1, type=int)
    parser.add_argument("--nb-iter2", default=20, type=int)
    parser.add_argument("--inv-mle", default=False, action="store_true")
    ## Shortest-Path Args
    parser.add_argument("-m", "--metric", type=str, help="define the scipy distance to be used (Default: euclidean or hamming for MSA)", default="euclidean")
    parser.add_argument("-x", "--matrix", help="if the input file contains already the complete upper triangle of a distance matrix (2 Formats: (idx_i idx_j distance) or simply distances list ) (Opt)", action="store_true")
    parser.add_argument("-k", "--n_neighbors", type=int, help="nearest_neighbors parameter (Default k=3)", default=3)
    parser.add_argument("-r", "--radius", type=float, help="use neighbor radius instead of nearest_neighbors  (Opt)", default=0.)
    parser.add_argument("-b", "--n_bins", type=int, help="number of bins for distance histogram (Default 50)", default=50)
    parser.add_argument("-M", "--r_max", default=0, type=float, help="fix the value of distance distribution maximum in the fit (Opt, -1 force the standard fit, avoiding consistency checks)")
    parser.add_argument("-n", "--r_min", default=-10, type=float, help="fix the value of shortest distance considered in the fit (Opt, -1 force the standard fit, avoiding consistency checks)")
    parser.add_argument("-D", "--direct", help="analyze the direct (not graph) distances (Opt)", action="store_true")
    parser.add_argument("-I", "--projection", help="produce an Isomap projection using the first ID components (Opt)", action="store_true")
    parser.add_argument("--cosine-dist", default=False, action="store_true")
    parser.add_argument("--first-n", default=None, type=int)

    args, _ = parser.parse_known_args()
    
    source_dir = source_dir if source_dir is not None else os.getcwd()
    
    if args.dataset == "all":
        args.dataset = ["CIFAR10", "MNIST", "CelebA"]
    else:
        args.dataset = [args.dataset]
        
    for dataset in args.dataset:
        logging.info(f"==================== Starting experiments for dataset: {dataset} ====================")
        suffix = f"{dataset}_class_{'-'.join(map(str, args.class_ind)) if args.class_ind is not None and len(args.class_ind) > 0 else 'all_classes'}"
        dest = os.path.join(source_dir, 'ID_ablation_results', suffix)
        os.makedirs(dest, exist_ok=True)
        
        real_images_root_dir = os.path.join(source_dir, 'data')
        
        if "celeba" not in dataset.lower():
            real_images_dir = os.path.join(real_images_root_dir, dataset, suffix, suffix)
        else:
            real_images_dir = os.path.join(real_images_root_dir, "CelebA", "imgs", "img_align_celeba")
        
        classes = args.class_ind if args.class_ind is not None and len(args.class_ind) > 0 else None
        dataloader = load_real_dataset(dataset=dataset, data_root=real_images_root_dir, cls=classes)
        real_images_dataset = dataloader.dataset
        num_images_to_generate = len(real_images_dataset)
        logging.info(f"Loaded real dataset: {dataset}, classes: {args.class_ind}, size: {num_images_to_generate}")
        # Remove from splits all values greater than the number of available images
        args.splits = [s for s in args.splits if s <= len(real_images_dataset)]
        if len(real_images_dataset) not in args.splits:
            args.splits.append(len(real_images_dataset))
        args.splits.append(len(real_images_dataset)//2)
        args.splits = sorted(args.splits)
        logging.info(f"Using splits: {args.splits} (out of {len(real_images_dataset)} available images)")
        target_intrinsic_dim = {s: [] for s in args.splits}
        execution_times = {s: [] for s in args.splits}
        
        # === Run the estimator for different percentages of real images ===
        for n_imgs in args.splits:
            for rep in range(args.repeat):
                set_seed(args.seed + rep)
                percentage = f"{n_imgs / len(real_images_dataset):.2f}"
                
                logging.info(f"==================== Starting repetition {rep+1}/{args.repeat} for {n_imgs} images (fraction {percentage}) ====================")
                
                real_images_dataset_tmp = torch.utils.data.Subset(real_images_dataset, np.random.choice(len(real_images_dataset), n_imgs, replace=False))
                start_time = time.time()
                results = run_estimator(args, 
                                        real_images_dataset_tmp, 
                                        verbose=True, 
                                        filename=os.path.join(dest, f"target_intrinsic_dim_results_{suffix}_r-{rep}_s-{n_imgs}.json"))
                end_time = time.time()
                target_intrinsic_dim[n_imgs].append(results['inv_mle_dim'])
                execution_times[n_imgs].append(end_time - start_time)

        # === Average the results over the N runs ===
        avg = [np.mean(target_intrinsic_dim[s]) for s in args.splits]
        std_dev = [np.std(target_intrinsic_dim[s]) for s in args.splits]
        
        avg_time = [np.mean(execution_times[s]) for s in args.splits]
        std_time = [np.std(execution_times[s]) for s in args.splits]
        # Save to a csv file: averages and std deviations for each split, number of images, avg time and std time
        with open(os.path.join(dest, f'intrinsic_dimension_{suffix}.csv'), 'w') as f:
            f.write('Split,Num_Images,Avg_Intrinsic_Dim,Std_Dev,Avg_Time(s),Std_Time(s)\n')
            for i, s in enumerate(args.splits):
                f.write(f'{int(len(real_images_dataset) / s)},{s},{avg[i]},{std_dev[i]},{avg_time[i]},{std_time[i]}\n')

        # === Plotting ===
        image_fractions = [f"{n_imgs/1000}k ({n_imgs / len(real_images_dataset):.2f})" for n_imgs in args.splits]
        plot_suffix = f"{suffix.replace('_', ' ')}"
        
        # Combined plot with dual y-axes
        fig, ax1 = plt.subplots()
        ax1.errorbar(image_fractions, avg, yerr=std_dev, fmt='-o', color='cornflowerblue', label='ID')
        ax1.set_xlabel('Number (Fraction) of Real Images')
        ax1.set_ylabel(f'Intrinsic Dimension ({args.repeat} repetitions)')
        ax1.tick_params(axis='y')
        ax1.tick_params(axis='x', rotation=20)
        ax1.set_title(f"Intrinsic Dimension vs Real Images from {plot_suffix}")
        
        # Add secondary y-axis for computation time
        ax2 = ax1.twinx()
        ax2.errorbar(image_fractions, avg_time, yerr=std_time, fmt='--', color='grey', label='Time')
        ax2.set_ylabel('Computation Time (s)')
        ax2.tick_params(axis='y')
        ax2.tick_params(axis='x', rotation=20)
        
        # Add legends for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Enlarge figure size for better readability
        fig.set_size_inches(7, 6)
        plt.tight_layout()
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(dest, f'intrinsic_dimension_{suffix}.png'), dpi=300)
        plt.close()
        print("Saved plot to", os.path.join(dest, f'intrinsic_dimension_{suffix}.png'))

# main(os.getcwd())