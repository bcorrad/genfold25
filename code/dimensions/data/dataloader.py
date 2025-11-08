import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
from .data import FileDataset, TensorDataset, read_hdf5
from . import unique_cifar_dataset as ucifar
from .icob_dataset_generator import generate_dataloader

# Resize all images to 32x32
RESIZE_SIZE = 32
resize_all = transforms.Lambda(lambda x: interpolate(torch.unsqueeze(x, 0), size=RESIZE_SIZE).squeeze())


def load_data(args, 
              train=True,
              dataset_split='train', 
              imagenet_uid_fp="imagenet_uid.npy", 
              data_root='./data',
              fonts_data_dir='/cmlscratch/pepope/public/gen_gan/fonts_data/fonts'):
    
    dataset_dict = {}
    gan_dataloader = None
    rand_inds = None

    if "icob" in args.dataset.lower():
        # Generate synthetic data if ICoB dataset samples do not exist
        if not os.path.exists(os.path.join(data_root, dataset_split)):
            print("Generating synthetic data for ICoB dataset")
            dataset_dict = generate_dataloader(dataDir=data_root, 
                                                imgResol=args.img_size, 
                                                batchSize=args.gan_batch_size, 
                                                nImages=args.n_images,
                                                split=dataset_split)

        else:
            for split in os.listdir(data_root):
                dataset_dict[split] = os.path.join(data_root, split)
            
        # Read the images from the folder
        try:
            dset = torchvision.datasets.ImageFolder(
                root=dataset_dict[dataset_split],
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    resize_all
                ]))
        except:
            # Take parent folder of dataset_dict[dataset_split]
            dset = torchvision.datasets.ImageFolder(
                root=os.path.dirname(dataset_dict[dataset_split]),
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    resize_all
                ]))
            
        # Pick args.max_num_samples samples from the dataset at random
        if args.max_num_samples != -1:
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=args.max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

        # dataloader (N, 3, 32, 32)
        gan_dataloader = DataLoader(dset, batch_size=args.gan_batch_size, shuffle=True)
    
    # Synthetic data must be loaded from `samples/`
    elif "samples" in args.dataset.lower():
       dset = FileDataset(args.dataset, getattr(args, 'max_num_samples', -1))

    elif "cifar" in args.dataset.lower():
        if args.dataset.lower() == "cifar100":
            dset_ = ucifar.CIFAR100
        else:
            dset_ = ucifar.CIFAR10

        dset = dset_(root=data_root,
                     train=train,
                     download=True,
                     transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),      # Source: https://github.com/kuangliu/pytorch-cifar/issues/19
                        resize_all,
                     ]), unique=True)

        if args.class_ind is not None and len(args.class_ind) > 0:
            print("Filtering class", args.class_ind)
            cls_inds = [i for i,x in enumerate(dset.targets) if x in args.class_ind]
            dset = torch.utils.data.Subset(dset, cls_inds)

        if args.max_num_samples != -1:
            print("Sampling", args.max_num_samples, "samples")
            # Randomly sample max_num_samples samples
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=args.max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

        gan_dataloader = DataLoader(dset, batch_size=args.gan_batch_size, shuffle=True)

    elif args.dataset.lower() == "svhn":
       if train:
           split = 'train'
       else:
           split = 'test'
       dset = torchvision.datasets.SVHN(root=data_root,
                                            split=split,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                # resize_all,
                                            ]))
       # Drop non-unique samples
       dset.data, uidx = np.unique(dset.data, return_index=True, axis=0)
       dset.labels = [dset.labels[i] for i in uidx]

       if args.class_ind != -1:
          cls_inds = [i for i,x in enumerate(dset.labels) if x == args.class_ind]
          dset = torch.utils.data.Subset(dset, cls_inds)

       if args.max_num_samples != -1:
          all_inds = np.arange(len(dset))
          rand_inds = np.random.choice(all_inds, size=args.max_num_samples, replace=False)
          dset = torch.utils.data.Subset(dset, rand_inds)

    elif args.dataset.lower() == "mnist":
        dset = torchvision.datasets.MNIST(data_root,
                                              download=True,
                                              train=train,
                                              transform=transforms.Compose([
                                                  transforms.Grayscale(3),
                                                  transforms.ToTensor(),
                                                  resize_all,
                                              ]))
        
        if args.class_ind is not None and len(args.class_ind) > 0:
            print("Filtering class", args.class_ind)
            cls_inds = [i for i,x in enumerate(dset.targets) if x in args.class_ind]
            dset = torch.utils.data.Subset(dset, cls_inds)

        if args.max_num_samples != -1:
            print("Sampling", args.max_num_samples, "samples")
            # Randomly sample max_num_samples samples
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=args.max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

        gan_dataloader = DataLoader(dset, batch_size=args.gan_batch_size, shuffle=True)

    elif args.dataset.lower() == "imagenet":
        #NB: we split on train. See comment below
        split = 'train'
        in_dir = os.path.join(args.imagenet_dir, split)
        dset = torchvision.datasets.ImageFolder(
                    in_dir,
                    transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        resize_all
                    ]))

        imagenet_uids = np.load(imagenet_uid_fp)
        keep_inds = imagenet_uids

        if args.class_ind != -1:
           cls_inds = [i for i,x in enumerate(dset.targets) if x == args.class_ind]
           keep_inds = [i for i in cls_inds if i in keep_inds]

        dset = torch.utils.data.Subset(dset, keep_inds)

        #NB: For ImageNet
        #There are 50  test  images per class, which is not enough.
        #There are 10k train images per class, which is more than enough.
        #So we split train/test from the training set
        #We **deterministically** split the test set to fix it for all runs
        current_state = np.random.get_state()
        np.random.seed(0)
        N = len(dset)
        all_inds = np.arange(N)
        np.random.shuffle(all_inds)
        dset = torch.utils.data.Subset(dset, all_inds)
        test_inds = all_inds[-args.num_test_per_cls:]
        train_inds = np.array([x for x in all_inds if x not in test_inds])
        #Downsample training set
        if args.max_num_samples != -1:
            train_inds = train_inds[:args.max_num_samples]
        test_dset = torch.utils.data.Subset(dset, test_inds)
        train_dset = torch.utils.data.Subset(dset, train_inds)
        #Reset state
        np.random.set_state(current_state)
        #Return specified split. Yes, only returning one at a time is wasteful
        #But otherwise it breaks the logic of this function and its caller
        if train:
            dset = train_dset
        else:
           dset = test_dset

    elif "fonts" in args.dataset.lower():
        if train:
            split = 'train'
        else:
            split = 'test'

        data_dir = fonts_data_dir

        #E.g. "fonts_1" = "fonts with 1/4 transformations"
        num_trans = args.dataset.split("_")[-1]
        #TODO: Don't hard-code filename pattern...
        fn_pattern = split + "_{}_" + "ABCDEFGHIJ_2000_6_28_dim={}.h5".format(num_trans)

        images_fn = fn_pattern.format("images")
        codes_fn = fn_pattern.format("codes")
        images_fp = os.path.join(data_dir, images_fn)
        codes_fp = os.path.join(data_dir, codes_fn)

        #Load images / classes
        images = read_hdf5(images_fp)
        codes = read_hdf5(codes_fp).astype(np.int)
        fonts = codes[:, 1]
        classes = codes[:, 2]

        #Cast images to uint8 (needed for PIL transform)
        images = np.uint8(images*255)

        #Drop non-unique samples
        images, uidx = np.unique(images, return_index=True, axis=0)
        classes = [classes[i] for i in uidx]

        #Convert to torch dataset
        X = [Image.fromarray(x) for x in images]
        Y = classes
        dset = TensorDataset((X,Y), transform=transforms.Compose([
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            resize_all
                                            ]))

        if args.class_ind != -1:
           cls_inds = [i for i,x in enumerate(dset.targets) if x == int(args.class_ind)]
           dset = torch.utils.data.Subset(dset, cls_inds)

        if args.max_num_samples != -1:
            all_inds = np.arange(len(dset))
            rand_inds = np.random.choice(all_inds, size=args.max_num_samples, replace=False)
            dset = torch.utils.data.Subset(dset, rand_inds)

    else:
        raise Exception("Dataset not understood")

    return dset, dataset_dict, gan_dataloader, rand_inds
