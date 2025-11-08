import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import gc  # Garbage collection
from torchvision.utils import save_image
from glob import glob

from config import GENERATED_IMAGES_FOLDER, CHECKPOINT_FOLDER, RESULTS_FOLDER, TRAINING_IMAGES_CHECKPOINT_FOLDER, INTERNAL_REPRESENTATIONS_FOLDER, CUDA as device, G_FEATURES_FILENAME

# Generator model with feature extraction
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=32*32*3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, img_size),
            nn.Tanh()
        )
        
        self.features = {}

    def forward(self, x, **kwargs):        
        self.features_dict = {}
        features = {}
        
        if 'store_features' in kwargs and kwargs['store_features']:
            for idx, layer in enumerate(self.model):
                if 'epoch' in kwargs: 
                    print(f"Saving features for epoch: {kwargs['epoch']}")
                    if idx == 0:
                        layer_name = f"input_epoch_{kwargs['epoch']}"
                        print(f"Epoch: {kwargs['epoch']} | Layer: {layer_name} | Shape: {x.shape}")
                        features[layer_name] = x.cpu().detach().numpy()
                    
                    layer_name = f"{idx}_{layer.__class__.__name__}"
                    print(f"Epoch: {kwargs['epoch']} | Layer: {layer_name} | Shape: {x.shape}")
                    x = layer(x)
                    features[layer_name] = x.cpu().detach().numpy()
            self.features_dict = features
            return x
        else:
            return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size=32*32*3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)

def save_generator_features(epoch, features, output_dir=INTERNAL_REPRESENTATIONS_FOLDER):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{G_FEATURES_FILENAME}_epoch_{epoch}.npy')
    np.save(output_path, features)
    print(f"Saved generator features for epoch {epoch} to {output_path}")
    
    # Clear caches and force garbage collection after saving features
    del features
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Run garbage collection to free memory

# # Function to save checkpoints of generator and critic
# def save_model_checkpoints(epoch, generator, discriminator, **kwargs):
#     if 'last_epoch' in kwargs and kwargs['last_epoch']:
#         epoch = 'last'
#     generator_chechpoint_folder = os.path.join(CHECKPOINT_FOLDER, "generator")
#     discriminator_checkpoint_folder = os.path.join(CHECKPOINT_FOLDER, "critic")
#     os.makedirs(generator_chechpoint_folder, exist_ok=True)
#     os.makedirs(discriminator_checkpoint_folder, exist_ok=True)
#     generator_checkpoint_path = os.path.join(generator_chechpoint_folder, f"generator_epoch_{epoch}.pth")
#     critic_checkpoint_path = os.path.join(discriminator_checkpoint_folder, f"critic_epoch_{epoch}.pth")
#     torch.save(generator.state_dict(), generator_checkpoint_path)
#     torch.save(discriminator.state_dict(), critic_checkpoint_path)

# Function to save checkpoints of generator and critic
def save_model_checkpoints(epoch, generator, discriminator, save_critic=False, **kwargs):
    if 'last_epoch' in kwargs and kwargs['last_epoch']:
        epoch = 'last'
    generator_chechpoint_folder = os.path.join(CHECKPOINT_FOLDER, "generator")
    os.makedirs(generator_chechpoint_folder, exist_ok=True)
    generator_checkpoint_path = os.path.join(generator_chechpoint_folder, f"generator_epoch_{epoch}.pth")
    torch.save(generator.state_dict(), generator_checkpoint_path)
    if save_critic:
        discriminator_checkpoint_folder = os.path.join(CHECKPOINT_FOLDER, "critic")
        os.makedirs(discriminator_checkpoint_folder, exist_ok=True)
        critic_checkpoint_path = os.path.join(discriminator_checkpoint_folder, f"critic_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), critic_checkpoint_path)

def save_generated_images(epoch, generated_images, output_dir=TRAINING_IMAGES_CHECKPOINT_FOLDER):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save as a grid
    if len(generated_images) > 16:
        generated_images = generated_images[:16]
    save_image(generated_images, os.path.join(output_dir, f'generated_images_epoch_{epoch}.png'), nrow=4, normalize=True)

def train(dataloader, latent_dim=100, n_epochs=100, save_interval=10, selected_epochs=None):
    
    if selected_epochs is None:
        selected_epochs = [epoch for epoch in range(n_epochs) if (epoch % save_interval == 0) or (epoch < 10) or (epoch > n_epochs / 2 - 5 and epoch < n_epochs / 2 + 5) or (epoch > n_epochs - 10)]
        selected_epochs.sort()
    
    print(f"Selected epochs: {selected_epochs}")
    
    # Initialize generator and discriminator, move to device (CPU or GPU)
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss().to(device)  # Loss moved to device        

    for epoch in range(n_epochs):
        batch_features = {}
        for batch_idx, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.view(imgs.size(0), -1).to(device)      # Move images to device
            z = torch.randn(imgs.size(0), latent_dim).to(device)    # Move latent vector to device

            # Update Discriminator
            optimizer_D.zero_grad()
            fake_imgs = generator(z).detach()
            real_loss = adversarial_loss(discriminator(real_imgs), torch.ones(imgs.size(0), 1).to(device))
            fake_loss = adversarial_loss(discriminator(fake_imgs), torch.zeros(imgs.size(0), 1).to(device))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad()
            if epoch in selected_epochs:# and batch_idx % 10 == 0:
                # Store features for the current batch
                gen_imgs = generator(z, store_features=True, epoch=epoch)
                batch_features[batch_idx] = generator.features_dict
            else:
                gen_imgs = generator(z, store_features=False, epoch=epoch)
            g_loss = adversarial_loss(discriminator(gen_imgs), torch.ones(imgs.size(0), 1).to(device))
            g_loss.backward()
            optimizer_G.step()

        if epoch in selected_epochs:
            save_generator_features(epoch, batch_features)
            save_generated_images(epoch, gen_imgs.view(-1, 3, 32, 32).cpu())  
            save_model_checkpoints(generator=generator, discriminator=discriminator, epoch=epoch, last_epoch=(epoch==n_epochs-1))

        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Clear caches and force garbage collection after each epoch
        torch.cuda.empty_cache()
        gc.collect()

def generate_images(generator=None, latent_dim=100, n_images=1, checkpoint_path=None, save_images_separately=False, map_location='cpu'):
    # If no checkpoint path is provided, load the last checkpoint
    if checkpoint_path is None:
        checkpoint_files = sorted(glob(os.path.join(CHECKPOINT_FOLDER, "*.pth")))
        if len(checkpoint_files) > 0:
            checkpoint_path = checkpoint_files[-1]
        else:
            raise FileNotFoundError("No checkpoints found in the folder.")
        
    # Extract the epoch number from the checkpoint path
    epoch = checkpoint_path.split('_')[-1].split('.')[0]

    # Load the generator and checkpoint, move to device
    generator = Generator(latent_dim=latent_dim)
    if map_location != 'cpu':
        generator = generator.cuda()
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    generator.load_state_dict(checkpoint)
    generator.eval()

    # Generate images, move latent vector to device
    z = torch.randn(n_images, latent_dim, device=map_location)
    gen_imgs = generator(z).view(n_images, 3, 32, 32).detach().cpu()  # Move generated images to CPU for saving

    # Save as a grid or single images based on the parameter
    if not os.path.exists(GENERATED_IMAGES_FOLDER):
        os.makedirs(GENERATED_IMAGES_FOLDER)
    
    # Reformat the images values for Matplotlib (Numpy array with values between 0 and 255), if needed
    if gen_imgs.min() < 0 or gen_imgs.max() > 1:
        gen_imgs = (gen_imgs + 1) / 2   # Scale back to [0, 1] range
    
    if save_images_separately is False:
        # Save a grid of images
        save_image(gen_imgs, os.path.join(RESULTS_FOLDER, f'generated_grid_{epoch}.png'), nrow=8)
    else:
        # Save individual images
        for idx, img in enumerate(gen_imgs):
            save_image(img, os.path.join(GENERATED_IMAGES_FOLDER, f'generated_image_{idx}.png'))

    return gen_imgs
