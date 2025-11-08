import os, sys, datetime, argparse, shutil

# Import the current path in the sys.path to import the distances module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from dimensions.data.dataloader import load_data
from utils import set_seed, set_gpu

calculate_metrics_before_training = False 
calculate_metrics_after_training = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--network", default="wgan_gp_conv", type=str, choices=["gan", "wgan_gp_linear", "dcgan", "wgan_gp_conv"], help="Network to use")
    parser.add_argument("--dataset", default="CelebA", type=str, choices=["CIFAR10", "MNIST", "ImageNet", "CelebA"], help="Dataset to use")
    parser.add_argument("--max_num_samples", default=10000, type=int, help="Maximum number of samples to process. Useful for evaluating convergence.")
    parser.add_argument("--separate-classes", default=False, action="store_true")
    parser.add_argument("--class-ind", default=[], type=list, help="class index to use for the dataset, if None or empty, all classes are used")

    # GAN Training args
    parser.add_argument("--n_epochs", default=400, type=int)
    parser.add_argument("--gan_batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=(32,32), type=int)
    parser.add_argument("--n_images", default=100000, type=int)
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--bsize", default=1024, type=int, help="batch size for previous images")
    parser.add_argument("--seed", default=1234, type=int)

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

args = parse_args()

print(args)

NETWORK = args.network
NETWORK = NETWORK.replace('_', '-')
N_EPOCHS = args.n_epochs

G_FEATURES_FILENAME = f"G_internal_representations"

SOURCE_PATH = None      # Path to the experiment folder: e.g., GenFold/20240916-104217/

CURRENT_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODELS_FOLDER = os.path.join(CURRENT_FILE_PATH, "models")

if SOURCE_PATH is None:
    # SOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    SOURCE_PATH = '../../../homeRepo/corradini'
    # os.makedirs(SOURCE_PATH, exist_ok=True)

    # Initialize the current timestamp for experiment folder
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Make dir of experiment appending the network, dataset, dataset_split (if dataset is icob), and timestamp
    EXPERIMENT_FOLDER = "experiments"
    EXPERIMENT_FOLDER = os.path.join(EXPERIMENT_FOLDER, f"network_{NETWORK}_dset_{args.dataset.lower()}")
    EXPERIMENT_FOLDER += f"_epochs_{N_EPOCHS}"
    if args.class_ind is not None:
        EXPERIMENT_FOLDER += f"_cls_{'-'.join(map(str, args.class_ind))}"
    EXPERIMENT_FOLDER += f"_dt_{TIMESTAMP}"
    os.makedirs(os.path.join(SOURCE_PATH, EXPERIMENT_FOLDER), exist_ok=True)
    print(f"Experiment folder: {EXPERIMENT_FOLDER}")
    try:
        # Save main.py to the experiment folder
        shutil.copy(__file__, os.path.join(SOURCE_PATH, EXPERIMENT_FOLDER, "config.py"))
    except Exception as e:
        print(f"Error copying files: {e}")
else:
    EXPERIMENT_FOLDER = SOURCE_PATH

CHECKPOINT_FOLDER = os.path.join(SOURCE_PATH, EXPERIMENT_FOLDER, "checkpoints")
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

RESULTS_FOLDER = os.path.join(SOURCE_PATH, EXPERIMENT_FOLDER, "results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

TRAINING_IMAGES_CHECKPOINT_FOLDER = os.path.join(RESULTS_FOLDER, "training_images")
os.makedirs(TRAINING_IMAGES_CHECKPOINT_FOLDER, exist_ok=True)

GENERATED_IMAGES_FOLDER = os.path.join(RESULTS_FOLDER, "generated_images")
os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)

INTERNAL_REPRESENTATIONS_FOLDER = os.path.join(RESULTS_FOLDER, "internal_representations")
os.makedirs(INTERNAL_REPRESENTATIONS_FOLDER, exist_ok=True)

PERSISTENT_HOMOLOGY_FOLDER = os.path.join(RESULTS_FOLDER, "persistent_homology")
os.makedirs(PERSISTENT_HOMOLOGY_FOLDER, exist_ok=True)

DATASET_FOLDER = "/repo/corradini/GenFold/data"

# Save the arguments to a json file
import json
with open(os.path.join(RESULTS_FOLDER, "args.json"), "w") as f:
    json.dump(vars(args), f)

# Force the GPU to be selected
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
CUDA = set_gpu(args.gpu)
set_seed(args.seed)

from dimensions.id_estimator import run_estimator
from dimensions.data.mydataloader import load_data
import os, shutil

# selected_epoch are between 0 and N_EPOCHS, with a step of 10, including the last epoch N_EPOCHS-1
selected_epochs = [epoch for epoch in range(N_EPOCHS) if (epoch % 10 == 0) or epoch == N_EPOCHS - 1]
dataset, dataset_dict, dataloader, dataset_rand_indices = load_data(args, data_root=DATASET_FOLDER)

if CALCULATE_METRICS_BEFORE_TRAINING:
    run_estimator(args, dataset, verbose=True, filename=os.path.join(RESULTS_FOLDER, f"intrinsic_dim_results.json"))

if NETWORK == "gan":
    import models.gan as gan    
    shutil.copy(f"{MODELS_FOLDER}/gan.py", os.path.join(RESULTS_FOLDER, "gan.py"))
    gan.train(dataloader,
              n_epochs=N_EPOCHS, 
              save_interval=10, 
              selected_epochs=selected_epochs)
    print("[TRAINING DONE]")

elif NETWORK == "dcgan":
    import models.dcgan as dcgan
    shutil.copy(f"{MODELS_FOLDER}/dcgan.py", os.path.join(RESULTS_FOLDER, "dcgan.py"))
    dcgan.train(dataloader,
                n_epochs=N_EPOCHS, 
                save_interval=10, 
                selected_epochs=selected_epochs)
    print("[TRAINING DONE]")

elif NETWORK == "wgan-gp-linear":
    import models.wgan_gp_linear as wgan_gp_linear
    shutil.copy(f"{MODELS_FOLDER}/wgan_gp_linear.py", os.path.join(RESULTS_FOLDER, "wgan_gp_linear.py"))
    wgan_gp_linear.train(dataloader,
                         n_epochs=N_EPOCHS, 
                         save_interval=10, 
                         selected_epochs=selected_epochs)
    print("[TRAINING DONE]")
    
elif NETWORK == "wgan-gp-conv":
    import models.wgan_gp_conv as wgan_gp_conv
    shutil.copy(f"{MODELS_FOLDER}/wgan_gp_conv.py", os.path.join(RESULTS_FOLDER, "wgan_gp_conv.py"))
    wgan_gp_conv.train(dataloader, 
                       n_epochs=N_EPOCHS, 
                       save_interval=10, 
                       selected_epochs=selected_epochs)
    print("[TRAINING DONE]")
    
if CALCULATE_METRICS_AFTER_TRAINING:
    gen_dataset, gen_dataset_dict, gen_dataloader, _ = load_data(args, data_root=GENERATED_IMAGES_FOLDER)
    run_estimator(args, gen_dataset, verbose=True, filename=os.path.join(RESULTS_FOLDER, f"gen_intrinsic_dim_results.json"))