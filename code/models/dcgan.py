import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

import gc # Garbage collection

from config import INTERNAL_REPRESENTATIONS_FOLDER, CHECKPOINT_FOLDER, TRAINING_IMAGES_CHECKPOINT_FOLDER, G_FEATURES_FILENAME

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: (3, 32, 32), Output: (64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (64, 16, 16), Output: (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Normalize activations for stability
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (128, 8, 8), Output: (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (256, 4, 4), Output: (512, 2, 2)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten the output from (512, 2, 2) to (512*2*2 = 2048)
            nn.Flatten(),

            # Fully connected layer: Input: (2048), Output: (1)
            nn.Linear(512 * 2 * 2, 1),

            # Sigmoid to get probability of being real or fake 
            # Remove if BCEWithLogitsLoss is used
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Fully connected layer: Input: (latent_dim), Output: (512*4*4 = 8192)
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape the output into a feature map: (512, 4, 4)
            nn.Unflatten(1, (512, 4, 4)),

            # Input: (512, 4, 4), Output: (256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),  # Normalize activations for stability
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (256, 8, 8), Output: (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (128, 16, 16), Output: (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: (64, 32, 32), Output: (3, 32, 32)
            nn.Conv2d(64, 3, kernel_size=3, padding=1),

            # Tanh to scale the output to the range [-1, 1]
            nn.Tanh()
        )
        
        self.features_dict = {}  # Dictionary to store features

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
    
    def get_internal_features(self):
        return self.features_dict

# Function to initialize model weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)

# Load CIFAR-10 dataset
def load_real_samples():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    return train_loader

# Generate real samples
def generate_real_samples(dataloader, n_samples):
    real_images, _ = next(iter(dataloader))
    real_images = real_images[:n_samples]
    labels = torch.ones(n_samples, 1)
    return real_images, labels

# Generate latent points
def generate_latent_points(latent_dim, n_samples):
    return torch.randn(n_samples, latent_dim).cuda()

# Generate fake samples
def generate_fake_samples(generator, latent_dim, n_samples):
    latent_points = generate_latent_points(latent_dim, n_samples)
    fake_images = generator(latent_points)
    labels = torch.zeros(n_samples, 1)
    return fake_images, labels

# Save a plot of generated images
def save_plot(examples, epoch, n=4, save_path='./'):
    # Normalize examples to [0, 1] range if examples are floats or negative through min-max scaling
    examples = (examples - examples.min()) / (examples.max() - examples.min())
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(np.transpose(examples[i].cpu().detach().numpy(), (1, 2, 0)))
    filename = os.path.join(save_path, f'{epoch}.png')
    plt.savefig(filename)
    plt.close()

# Summarize performance
def summarize_performance(epoch, generator, discriminator, dataloader, latent_dim, n_samples=150):
    real_images, real_labels = generate_real_samples(dataloader, n_samples)
    real_acc = discriminator(real_images.cuda()).mean().item()

    fake_images, fake_labels = generate_fake_samples(generator, latent_dim, n_samples)
    fake_acc = discriminator(fake_images).mean().item()

    print(f'>Accuracy real: {real_acc*100:.0f}%, fake: {fake_acc*100:.0f}%')
    save_plot(fake_images, epoch, n=4, save_path=TRAINING_IMAGES_CHECKPOINT_FOLDER)

# Function to save features as a single .npy file
def save_features_as_single_npy(epoch, features_per_batch, **kwargs):
    if 'batch' in kwargs:
        batch = kwargs['batch']
        npy_save_path = os.path.join(INTERNAL_REPRESENTATIONS_FOLDER, f"{G_FEATURES_FILENAME}_epoch_{epoch}_batch_{batch}.npy")
    else:
        npy_save_path = os.path.join(INTERNAL_REPRESENTATIONS_FOLDER, f"{G_FEATURES_FILENAME}_epoch_{epoch}.npy")
    np.save(npy_save_path, features_per_batch)
    print(f"[SAVED] Features for epoch {epoch} saved to {npy_save_path}")
    
    # Clear caches and force garbage collection after saving features
    del features_per_batch
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

# Updated train function with parameters
def train(dataloader, 
          latent_dim=100, 
          n_epochs=300, 
          lr_g=0.0002, 
          lr_d=0.0002, 
          save_interval=10, 
          selected_epochs=None):

    # Initialize models
    generator = Generator(latent_dim).cuda()
    discriminator = Discriminator().cuda()

    # Apply weight initialization
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Select epochs to save features ((epoch%save_interval==0) or (epoch<10) or (epoch>n_epochs/2-5 and epoch<n_epochs/2+5) or (epoch>n_epochs-10))
    if selected_epochs is None:
        selected_epochs = [epoch for epoch in range(n_epochs) if (epoch % save_interval == 0) or (epoch < 10) or (epoch > n_epochs/2-5 and epoch < n_epochs/2+5) or (epoch > n_epochs-10)]
    
    for epoch in range(n_epochs):
        batch_features = {}  # Store features for all batches in an epoch
        for batch_idx, (real_images, _) in enumerate(dataloader):
            # print(f'Batch [{i+1}/{len(dataloader)}]')
            real_images = real_images.cuda()
            n_samples = real_images.size(0)
            real_labels = torch.ones(n_samples, 1).cuda()

            fake_images, fake_labels = generate_fake_samples(generator, latent_dim, n_samples)
            fake_labels = fake_labels.cuda()

            # Update Discriminator
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_images), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Update Generator
            optimizer_g.zero_grad()
            latent_points = generate_latent_points(latent_dim, n_samples)
            generated_labels = torch.ones(n_samples, 1).cuda()
            # g_loss = criterion(
            #     discriminator(
            #         generator(latent_points, store_features=(epoch in selected_epochs and batch_idx % 10 == 0), epoch=epoch)), 
            #         generated_labels
            #     )
            
            if epoch in selected_epochs and batch_idx % 10 == 0:
                # Store features for the current batch
                gen_imgs = generator(latent_points, store_features=True, epoch=epoch)
                batch_features[batch_idx] = generator.features_dict
            else:
                gen_imgs = generator(latent_points, store_features=False, epoch=epoch)
                
            g_loss = criterion(discriminator(gen_imgs), generated_labels)
                
            g_loss.backward()
            optimizer_g.step()
            
            # # Save the generator features every 10 batches
            # if epoch in selected_epochs and batch_idx % 10 == 0:
            #     batch_features[batch_idx] = generator.get_internal_features()

        print(f'Epoch [{epoch}/{n_epochs}] | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}')
        
        if epoch in selected_epochs:# or epoch == n_epochs-1:
            summarize_performance(epoch, generator, discriminator, dataloader, latent_dim)
            save_features_as_single_npy(epoch, batch_features)
            save_model_checkpoints(generator=generator, discriminator=discriminator, epoch=epoch, last_epoch=(epoch==n_epochs-1))

        # Clear caches and force garbage collection after each epoch
        torch.cuda.empty_cache()
        gc.collect()


# Function to generate images from a checkpoint
def generate_images(checkpoint_path, latent_dim=100, n_images=16, image_size=(32, 32), save_path='generated_images.png', save_images_separately=False, map_location='cpu'):
        
    # Load the generator model
    generator = Generator(latent_dim)
    if map_location != 'cpu':
        generator = generator.to(map_location)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    generator.eval()

    # Generate latent points
    latent_points = torch.randn(n_images, latent_dim, device=map_location)

    # Generate images
    with torch.no_grad():
        fake_images = generator(latent_points)
    
    # Scale images from [-1,1] to [0,1]
    if np.min(fake_images.cpu().detach().numpy()) < 0 or np.max(fake_images.cpu().detach().numpy()) > 1:
        fake_images = (fake_images + 1) / 2.0

    if not save_images_separately:
        # Create a grid of images and save
        grid = vutils.make_grid(fake_images.cpu(), nrow=int(np.sqrt(n_images)), padding=2, normalize=True)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(grid, (1,2,0)))
        
        # Save generated images
        plt.savefig(save_path)
        plt.close()

        print(f'Generated images saved to {save_path}')

    if save_images_separately:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save every image separately
        for i in range(n_images):
            img = fake_images[i].cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imsave(os.path.join(save_path, f'generated_image_{i}.png'), img)

        print(f'Generated images saved separately to the {save_path} directory')

