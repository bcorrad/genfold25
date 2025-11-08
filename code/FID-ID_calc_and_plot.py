import sys, argparse, os
# Import the current path in the sys.path to import the distances module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dimensions.id_estimator import run_estimator
from config import args
# Read images in real_images_dir and return the dataset. Use torch ImageFolder to load the images
from torchvision.datasets import ImageFolder 
from ID_ablation import load_real_dataset  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--network", default="dcgan", type=str, choices=["gan", "wgan_gp_linear", "dcgan", "wgan_gp_conv"], help="Network to use")
    parser.add_argument("--dataset", default="MNIST", type=str, choices=["CIFAR10", "MNIST", "ImageNet", "ICoB"], help="Dataset to use")
    parser.add_argument("--icob-split", default="monochrome", type=str, choices=["monochrome", "chessboards", "single_shapes", "multi_shapes"], help="ICoB split to use.")
    parser.add_argument("--anchor-samples", default=0, type=int, help="0 for using all samples from the training set")
    parser.add_argument("--anchor-ratio", default=0, type=float, help="0 for using all samples from the training set")
    parser.add_argument("--max_num_samples", default=-1, type=int, help="Maximum number of samples to process. Useful for evaluating convergence.")
    parser.add_argument("--imagenet-dir", default="/fs/cml-datasets/ImageNet/ILSVRC2012/train", type=str)
    parser.add_argument("--n_cls", default=None, type=int, help="A redundant flag for specifying number of classes.")
    parser.add_argument("--separate-classes", default=False, action="store_true")
    parser.add_argument("--class-ind", default=[3], type=list, help="class index to use for the dataset, if None or empty, all classes are used")

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
    return args

# Load GAN checkpoint
def load_gan_model(checkpoint_path, generator, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()
    return generator

# Generate images from the GAN model
def generate_images(generator, num_images_to_generate, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    generator.to(device)
    z_dim = 100  # Assuming z_dim for the latent vector
    for i in range(num_images_to_generate):
        z = torch.randn(1, z_dim).to(device)  # Random latent vector
        with torch.no_grad():
            generated_image = generator(z).squeeze(0).cpu()
        generated_image = generated_image.reshape(3, 32, 32)
        # Denormalize the image
        generated_image = (generated_image + 1) / 2
        img_path = os.path.join(output_dir, f"generated_{i}.png")
        transforms.ToPILImage()(generated_image).save(img_path)
        
        
# Calculate FID score
def calculate_fid(real_dir, generated_dir):
    fid_value = fid_score.calculate_fid_given_paths([real_dir, generated_dir], batch_size=32, device='cuda', dims=2048)
    return fid_value

# Main execution
if __name__ == '__main__':
    args = parse_args()
    remove_checkpoint_folder = False  # Remove the checkpoint folder after zipping
    network_target = None  # Target network to calculate FID scores
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # experiments = '/repo/corradini/GenFold/code/experiments'  # Path to the experiments folder
    experiments = "../../../homeRepo/corradini/experiments/"
    real_images_root_dir = '/repo/corradini/GenFold/data'
    
    num_images_to_generate = 3000   # Number of images to generate for FID calculation if otherwise specified

    for experiment in os.listdir(experiments):
        if "dcgan_dset_celeba" not in experiment.lower():
            print(f"Skipping experiment: {experiment}")
            continue
        calculate_FID = True               # Calculate FID scores
        calculate_ID = True                # Calculate Intrinsic Dimension
        PLOT_FID = True                     # Plot FID scores
        PLOT_ID = True                      # Plot Intrinsic Dimension progression
        dataloader = None
        if os.path.isdir(os.path.join(experiments, experiment)):
            
            os.makedirs(os.path.join(experiments, experiment, 'results', 'FID'), exist_ok=True)
            os.makedirs(os.path.join(experiments, experiment, 'results', 'ID'), exist_ok=True)
            os.makedirs(os.path.join(experiments, experiment, 'results', 'FID_generated_images'), exist_ok=True)
            
            print(f"Processing experiment: {os.path.join(experiments, experiment)}")
            epoch_gen_intrinsic_df = pd.DataFrame(columns=['epoch', 'intrinsic_dim'])
            FID_df = pd.DataFrame(columns=['epoch', 'FID'])

            # Parse the experiment path to get the network, dataset, epochs, and classes
            network = experiment.split('network_')[1].split('_dset')[0]
            dataset = experiment.split('dset_')[1].split('_epochs')[0]
            if "celeba" in dataset.lower():
                dataset = "CelebA"
            epochs = experiment.split('_epochs_')[1].split('_cls')[0]
            cls = experiment.split('_cls_')[1].split('_dt')[0].split('-')
                    
            fid_val_path = glob.glob(os.path.join(experiments, experiment, 'results', 'FID', f'fid_scores_*.csv'))
            if len(fid_val_path) > 0:
                fid_val_path = fid_val_path[0]
                fid_plot_path = fid_val_path.replace('.csv', '.png')
                print(f"Found FID scores file: {fid_val_path}. Skipping...")
            else:
                calculate_FID = True
            
            # fid_plot_path = fid_val_path.replace('.csv', '.png')
            id_progression_path = glob.glob(os.path.join(experiments, experiment, 'results', 'ID', f'intrinsic_dim_results_{dataset}_*.csv'))
            if len(id_progression_path) > 0:
                id_progression_path = id_progression_path[0]
                id_progression_plot_path = id_progression_path.replace('.csv', '.png')
                print(f"Found Intrinsic Dimension progression file: {id_progression_path}. Skipping...")
            else:
                calculate_ID = True

            if calculate_ID or calculate_FID:
                checkpoint_path = os.path.join(experiments, experiment, 'checkpoints', 'generator')
                checkpoints = glob.glob(os.path.join(checkpoint_path, '*.pth'))
                
                if len(checkpoints) == 0:
                    print(f"No checkpoints found for {experiment}. Skipping...")
                    continue            
            
                # Save the real images for FID calculation in a separate folder
                dataset_root = os.path.join(real_images_root_dir, dataset)
                real_images_dir = os.path.join(real_images_root_dir, dataset, '-'.join(map(str, cls)), f"{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}")
                
                if os.path.exists(real_images_dir) and len(os.listdir(real_images_dir)) > 0:
                    print(f"Real images directory already exists: {real_images_dir}, containing {num_images_to_generate} images")
                else:
                    os.makedirs(real_images_dir, exist_ok=True)
                    dataloader = load_real_dataset(dataset=dataset, data_root=real_images_root_dir, cls=[], max_num_samples=0.1)
                    num_images_to_generate = len(dataloader.dataset)
                    print(f"Loaded real dataset: {dataset}, classes: {cls}, size: {num_images_to_generate}")
                    for i, (imgs, _) in enumerate(dataloader):
                        for j, img in enumerate(imgs):
                            img_path = os.path.join(real_images_dir, f'real_{i}_{j}.png')
                            transforms.ToPILImage()(img).save(img_path)
                        
                fid_val_path = os.path.join(experiments, experiment, 'results', 'FID', f"fid_scores_{dataset}_{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}_{num_images_to_generate}.csv")
                fid_plot_path = fid_val_path.replace('.csv', '.png')
                
                id_progression_path = os.path.join(experiments, experiment, 'results', 'ID', f"intrinsic_dim_results_{dataset}_{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}_{num_images_to_generate}.csv")
                id_progression_plot_path = id_progression_path.replace('.csv', '.png')
                    
                real_images_dataset = ImageFolder(os.path.dirname(real_images_dir)+"/images/", transform=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Resize((32, 32))
                                                                    ]))

                # Print the correspondence between class indices and class names
                print(real_images_dataset.class_to_idx)
                # Take 25% of the real images for intrinsic dimension calculation. Keep it in ImageFolder format
                real_images_dataset = torch.utils.data.Subset(real_images_dataset, np.random.choice(len(real_images_dataset), int(len(real_images_dataset) * 0.01), replace=False))
                print(f"Calculating Intrinsic Dimension for real images on {len(real_images_dataset) * 0.01} images")
                
                results = run_estimator(args, real_images_dataset, verbose=True, filename=os.path.join(experiments, experiment, 'results', 'ID', f"target_intrinsic_dim_results_{dataset}_{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}.json"))
                real_intrinsic_dim = results['inv_mle_dim']
                
                # Read all the checkpoints in the folder
                checkpoints = glob.glob(os.path.join(checkpoint_path, '*.pth')) 
                # Sort the checkpoints based on the epoch number. If the epoch is "last", replace it with 399
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_epoch_')[-1].split('.')[0]) if 'last' not in x else 399)
            
                for checkpoint_path in checkpoints:                
                    # Initialize the correct GAN model based on the parsed network type
                    if network == 'dcgan':
                        from models.dcgan import Generator as YourGANGeneratorModel
                    elif network == 'wgan-gp-linear':
                        from models.wgan_gp_linear import Generator as YourGANGeneratorModel
                    elif network == 'wgan-gp-conv':
                        from models.wgan_gp_conv import Generator as YourGANGeneratorModel
                    elif network == 'gan':
                        from models.gan import Generator as YourGANGeneratorModel
                        
                    epoch = checkpoint_path.split('_epoch_')[-1].split('.')[0]
                    if 'last' in epoch:
                        epoch = '399'
                        
                    generated_images_dir = os.path.join(experiments, experiment, 'results', 'FID_generated_images', f'{network}_{dataset}_epochs_{epochs}_cls_{cls}', epoch, 'images')
                    os.makedirs(generated_images_dir, exist_ok=True)
                    print(f"Generating {num_images_to_generate} images for {experiment} epoch {epoch}")

                    # Initialize your generator model here
                    generator = YourGANGeneratorModel()

                    # Load the trained generator
                    generator = load_gan_model(checkpoint_path, generator, device)

                    # Generate images using the GAN
                    generate_images(generator, num_images_to_generate, generated_images_dir, device)

                    if calculate_FID:
                        # Calculate FID score between real and generated images
                        fid_value = calculate_fid(real_images_dir+"images/0/img_align_celeba", generated_images_dir)
                        print(f'FID score for {experiment} epoch {epoch}: {fid_value}')
                        FID_df.loc[len(FID_df)] = {'epoch': int(epoch), 'FID': fid_value}
                        # Save the FID scores to a CSV file
                        FID_df.to_csv(fid_val_path, index=False)
                        
                    if calculate_ID:
                        generated_images_dataset = ImageFolder(os.path.join(experiments, experiment, 'results', 'FID_generated_images', f'{network}_{dataset}_epochs_{epochs}_cls_{cls}', epoch), 
                                                                transform=transforms.Compose([
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Resize((32, 32))
                                                                                            ]))
                        # Take 50% of the generated images for intrinsic dimension calculation
                        generated_images_dataset = torch.utils.data.Subset(generated_images_dataset, np.random.choice(len(generated_images_dataset), int(len(generated_images_dataset) * 0.25), replace=False))
                        print(f"Calculating Intrinsic Dimension for {experiment} epoch {epoch} on {len(generated_images_dataset)} images")
                        epochs_results = run_estimator(args,
                                                        generated_images_dataset,
                                                        verbose=True, 
                                                        filename=os.path.join(experiments, experiment, 'results', 'ID', f"intrinsic_dim_results_{dataset}_{epoch}_{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}.json"))
                        epoch_intrinsic_dim = epochs_results['inv_mle_dim']
                        epoch_gen_intrinsic_df.loc[len(epoch_gen_intrinsic_df)] = {'epoch': int(epoch), 'intrinsic_dim': epoch_intrinsic_dim}
                        epoch_gen_intrinsic_df.to_csv(id_progression_path, index=False)
                    
                    # Remove files from the generated images directory (keep only 10 images)
                    generated_images = os.listdir(generated_images_dir)
                    for img in generated_images[15:]:
                        os.remove(os.path.join(generated_images_dir, img))
            
            if PLOT_FID:   
                # Read the FID scores from fid_val_path
                FID_df = pd.read_csv(fid_val_path)
                
                FID_df['epoch'] = FID_df['epoch'].replace('last', 399)
                FID_df['epoch'] = FID_df['epoch'].astype(int)
                FID_df_sorted = FID_df.sort_values(by='epoch')
                FID_df.to_csv(fid_val_path, index=False)
                print(f"Saved FID scores for {experiment} to {fid_val_path}")
                
                if remove_checkpoint_folder:
                    # Delete the checkpoint folder if the zip file was created successfully
                    if os.path.exists(os.path.join(experiments, experiment, f'checkpoints{experiment}.zip')):
                        # os.system(f"rm -r {os.path.join(experiments, experiment, 'checkpoints')}")
                        print(f"Zipped and deleted checkpoints folder for {experiment}")
                    else:
                        print(f"Failed to zip checkpoints folder for {experiment}")
                
                # Plotting epoch vs FID with correct labels from the epoch column on x-axis and a longer figure
                plt.figure(figsize=(16, 8))
                plt.plot(range(len(FID_df['epoch'])), FID_df['FID'], marker='o', linestyle='-')
                plt.xlabel('epoch')
                plt.ylabel('FID Score')
                title = f"[{network.upper()} on {dataset.upper()} {'class ' + ''.join(map(str, cls)) if len(cls) == 1 else 'classes ' + ', '.join(map(str, cls))}] FID Progression through epochs".replace('FID Progression', r'$\bf{FID}$')
                
                plt.title(title, fontsize=16)
                plt.xticks(ticks=range(len(FID_df['epoch'])), labels=FID_df['epoch'], rotation=90)  # Set evenly spaced ticks with epoch labels
                plt.grid(True)
                plt.savefig(fid_plot_path, dpi=600)
                plt.close()
                print(f"[FID PLOT SAVED] Saved FID plot for {experiment} to {fid_plot_path}")
                
            if PLOT_ID:
                # Read the Intrinsic Dimension progression from id_progression_path
                epoch_gen_intrinsic_df = pd.read_csv(id_progression_path)
                
                    
                # real_images_dir parent directory using os.path.dirname
                real_images_dir = os.path.join(real_images_root_dir, dataset, '-'.join(map(str, cls)), f"{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}")
                
                real_images_dataset = ImageFolder(os.path.dirname(real_images_dir)+"/images/", transform=transforms.Compose([
                                                                        transforms.ToTensor(),
                                                                        transforms.Resize((32, 32))
                                                                    ]))
                # Print the correspondence between class indices and class names
                print(real_images_dataset.class_to_idx)
                # Take 25% of the real images for intrinsic dimension calculation. Keep it in ImageFolder format
                real_images_dataset = torch.utils.data.Subset(real_images_dataset, np.random.choice(len(real_images_dataset), int(len(real_images_dataset) * 0.25), replace=False))
                print(f"Calculating Intrinsic Dimension for real images on {len(real_images_dataset)} images")
                
                results = run_estimator(args, real_images_dataset, verbose=True, filename=os.path.join(experiments, experiment, 'results', 'ID', f"target_intrinsic_dim_results_{dataset}_{'-'.join(map(str, cls)) if len(cls) > 1 else ''.join(map(str, cls))}.json"))
                target_intrinsic_dim = results['inv_mle_dim']
                
                epoch_gen_intrinsic_df = epoch_gen_intrinsic_df.replace('last', 399)
                epoch_gen_intrinsic_df['epoch'] = epoch_gen_intrinsic_df['epoch'].astype(int)
                epoch_gen_intrinsic_df = epoch_gen_intrinsic_df.sort_values(by='epoch')
                
                epoch_gen_intrinsic_df.to_csv(id_progression_path, index=False)
                print(f"Saved Intrinsic Dimension progression for {experiment} to {id_progression_path}")
                
                # Plot intrinsic dimension progression
                plt.figure(figsize=(16, 8))
                plt.plot(range(len(epoch_gen_intrinsic_df['epoch'])), epoch_gen_intrinsic_df['intrinsic_dim'], marker='o', linestyle='-')
                # Plot the target intrinsic dimension
                plt.axhline(y=target_intrinsic_dim, color='r', linestyle='--', label='Target Intrinsic Dimension')
                plt.xlabel('epoch')
                plt.ylabel('Intrinsic Dimension')
                
                title = f"[{network.upper()} on {dataset.upper()} {'class ' + ''.join(map(str, cls)) if len(cls) == 1 else 'classes ' + ', '.join(map(str, cls))}] Intrinsic Dimension Progression through epochs".replace('Intrinsic Dimension Progression', r'$\bf{Intrinsic\ Dimension\ Progression}$')
                plt.title(title)
                
                plt.xticks(ticks=range(len(epoch_gen_intrinsic_df['epoch'])), labels=epoch_gen_intrinsic_df['epoch'], rotation=90)  # Set evenly spaced ticks with epoch labels
                plt.grid(True)
                
                plt.savefig(id_progression_plot_path, dpi=600)
                plt.close()
                print(f"[ID PLOT SAVED] Saved Intrinsic Dimension progression plot for {experiment} to {id_progression_plot_path}")
                
            print(f"[DONE] Processed experiment: {experiment}")
            
    print("[DONE] FID calculation completed")
            