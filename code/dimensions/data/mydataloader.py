import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
from .data import TensorDataset, read_hdf5
from . import unique_cifar_dataset as ucifar

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

    if "cifar" in args.dataset.lower():
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

    elif "celeba" in args.dataset.lower():
        import subprocess
        from torchvision.datasets.folder import default_loader

        celeba_root = os.path.join(data_root, "CelebA")
        img_dir = os.path.join(celeba_root, "img_align_celeba")
        partition_file = os.path.join(celeba_root, "Eval", "list_eval_partition.txt")

        # Download and extract if missing
        if not os.path.isdir(img_dir) or not os.path.isfile(partition_file):
            print("[INFO] CelebA not found locally. Downloading...")
            os.makedirs(celeba_root, exist_ok=True)
            celeba_zip = os.path.join(celeba_root, "celeba.zip")
            if not os.path.isfile(celeba_zip):
                subprocess.run(["wget", "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip", "-O", celeba_zip], check=False)
            print("[INFO] Extracting CelebA...")
            subprocess.run(["unzip", "-o", celeba_zip, "-d", celeba_root], check=False)
            print("[INFO] Extraction completed.")

        # Parse the partition file (0=train, 1=val, 2=test)
        partitions = {}
        if os.path.isfile(partition_file):
            with open(partition_file, "r", encoding="utf-8", errors="ignore") as pf:
                for line in pf:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    ls = s.lower()
                    if "<html" in ls or "</html>" in ls or "<head" in ls or "<body" in ls:
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

        # Choose split
        split_id = 0 if train else 2  # 0:train, 2:test
        selected_files = [f for f, sid in partitions.items() if sid == split_id] if partitions else []
        if not selected_files:
            if os.path.isdir(img_dir):
                selected_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))] 
            else:
                raise RuntimeError("[ERROR] CelebA images directory not found and partition file parsing failed.")
            
        # Randomly select max_num_samples if specified
        if args.max_num_samples != -1 and len(selected_files) > args.max_num_samples:
            selected_files = np.random.choice(selected_files, size=args.max_num_samples, replace=False).tolist()   

        celeba_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor()
        ])

        class CelebADataset(torch.utils.data.Dataset):
            def __init__(self, img_dir, file_list, transform=None):
                self.img_dir = img_dir
                self.file_list = file_list
                self.transform = transform

            def __len__(self):
                return len(self.file_list)

            def __getitem__(self, idx):
                fname = self.file_list[idx]
                path = os.path.join(self.img_dir, fname)
                img = default_loader(path)
                if self.transform:
                    img = self.transform(img)
                return img

        dset = CelebADataset(img_dir, selected_files, transform=celeba_transform)
        # Calculate targets as 0
        Y = [0]*len(dset)
        # Keep batch size and loader settings consistent with other datasets if available on args
        num_workers = getattr(args, "num_workers", max(1, (os.cpu_count() or 2)//2))
        pin = torch.cuda.is_available()
        bs = getattr(args, "batch_size", 32)
        
        gan_dataloader = TensorDataset((dset, Y), transform=None)
        gan_dataloader = DataLoader(gan_dataloader, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin)

    else:
        raise Exception("Dataset not understood")

    return dset, dataset_dict, gan_dataloader, rand_inds
