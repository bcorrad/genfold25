import torch
import torch.nn as nn
from torchvision import utils
import os
import numpy as np
import gc

# Assume the following folder paths are defined in config.py
from config import INTERNAL_REPRESENTATIONS_FOLDER, CHECKPOINT_FOLDER, TRAINING_IMAGES_CHECKPOINT_FOLDER, G_FEATURES_FILENAME, CUDA as device

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, channels=3, img_size=32, latent_dim=100):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        self.features_dict = {}  # Dictionary to store features

    def forward(self, z, **kwargs):
        features = {}
        self.features_dict = {}
        
        is_input = True
        
        if 'store_features' in kwargs and kwargs['store_features']:
            if 'epoch' in kwargs:
                print(f"Saving features for epoch: {kwargs['epoch']}")
                if is_input:  # Save input features
                    layer_name = f"input_epoch_{kwargs['epoch']}"
                    print(f"Epoch: {kwargs['epoch']} | Layer: {layer_name} | Shape: {z.shape}")
                    features[layer_name] = z.cpu().detach().numpy()
                    is_input = False
                    
                # Save features for l1 layer
                out = self.l1(z)
                out = out.view(out.shape[0], 128, self.init_size, self.init_size)
                layer_name = f"{0}_{self.l1[0].__class__.__name__}"
                print(f"Epoch: {kwargs['epoch']} | Layer: {layer_name} | Shape: {out.shape}")
                features[layer_name] = out.detach().cpu().numpy()
                
                # Save features for conv_blocks
                for idx, layer in enumerate(self.conv_blocks):
                    out = layer(out)
                    # layer_type = layer.__class__.__name__
                    layer_name = f"{idx+1}_{layer.__class__.__name__}"
                    print(f"Epoch: {kwargs['epoch']} | Layer: {layer_name} | Shape: {out.shape}")
                    features[layer_name] = out.detach().cpu().numpy()
                        
                self.features_dict = features
        else:
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            out = self.conv_blocks(out)
            
        # img = out
        return out

    def get_features(self):
        return self.features_dict

# Define the Critic class (replaces Discriminator in WGAN-GP)
class Critic(nn.Module):
    def __init__(self, channels, img_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512 * (img_size // 16) ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.fc(out)
        return validity

# Function for Gradient Penalty
def gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Function to save generated images using torch
def save_generated_images(epoch, generator, fixed_noise, save_interval):
    generator.eval()  
    with torch.no_grad():
        gen_imgs = generator(fixed_noise).detach().cpu()
        gen_imgs = (gen_imgs + 1) / 2  # Denormalize images from [-1, 1] to [0, 1]
    if len(gen_imgs) > 16:
        gen_imgs = gen_imgs[:16]
    save_path = os.path.join(TRAINING_IMAGES_CHECKPOINT_FOLDER, f"epoch_{epoch}.png")
    utils.save_image(gen_imgs, save_path, nrow=4, normalize=True)

# Function to save features as a single .npy file
def save_features_as_single_npy(epoch, features):
    npy_save_path = os.path.join(INTERNAL_REPRESENTATIONS_FOLDER, f"{G_FEATURES_FILENAME}_epoch_{epoch}.npy")
    np.save(npy_save_path, features)
    print(f"[SAVED] Features for epoch {epoch} saved to {npy_save_path}")
    
    # Clear caches and force garbage collection after saving features
    del features
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Run garbage collection to free memory

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

# WGAN-GP Training Function
def train(dataloader, latent_dim=100, channels=3, img_size=32, save_interval=10, n_epochs=300, lambda_gp=10, n_critic=5, selected_epochs=None):
    if selected_epochs is None:
        selected_epochs = [epoch for epoch in range(n_epochs) if (epoch % save_interval == 0) or (epoch < 10) or (epoch > n_epochs / 2 - 5 and epoch < n_epochs / 2 + 5) or (epoch > n_epochs - 10)]
        selected_epochs.sort()
        
    print(f"Selected epochs: {selected_epochs}")
    generator = Generator(latent_dim=latent_dim, channels=channels, img_size=img_size).to(device)
    critic = Critic(channels, img_size).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

    fixed_noise = torch.randn(16, latent_dim, device=device)
    
    for epoch in range(n_epochs):
        batch_features = {}  # Store features for all batches in an epoch

        for batch_idx, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # Train Critic
            optimizer_C.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(z).detach()
            real_validity = critic(real_imgs)
            fake_validity = critic(fake_imgs)
            gp = gradient_penalty(critic, real_imgs, fake_imgs)
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
            c_loss.backward()
            optimizer_C.step()

            # Train Generator every n_critic steps
            if batch_idx % n_critic == 0:
                optimizer_G.zero_grad()
                
                z = torch.randn(imgs.size(0), latent_dim, device=device)
                if epoch in selected_epochs: # and batch_idx % 10 == 0:
                    # Store features for the current batch
                    gen_imgs = generator(z, store_features=True, epoch=epoch)
                    batch_features[batch_idx] = generator.features_dict
                else:
                    gen_imgs = generator(z, store_features=False, epoch=epoch)
                    
                g_loss = -torch.mean(critic(gen_imgs))
                g_loss.backward()
                optimizer_G.step()

        # Save generated images at intervals
        if epoch in selected_epochs:
            save_features_as_single_npy(epoch, batch_features)
            save_generated_images(epoch, generator, fixed_noise, save_interval)
            save_model_checkpoints(generator=generator, discriminator=critic, epoch=epoch, last_epoch=(epoch==n_epochs-1))
            
        print(f"[Epoch {epoch}/{n_epochs}] [Critic loss: {c_loss.item()}] [Generator loss: {g_loss.item()}]")
        
        # Clear caches and force garbage collection after each epoch
        torch.cuda.empty_cache()
        gc.collect()

# Function to generate images from a specific checkpoint
# from config import GENERATED_IMAGES_FOLDER
def generate_images(checkpoint_path, save_path=".", save_images_separately=False, latent_dim=100, channels=3, img_size=32, n_images=16, map_location='cpu'):
    generator = Generator(latent_dim, channels, img_size)
    if map_location != 'cpu':
        generator = generator.to(map_location)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    generator.eval()
    fixed_noise = torch.randn(n_images, latent_dim, device=map_location)
    with torch.no_grad():
        generated_images = generator(fixed_noise).detach().cpu()
        generated_images = (generated_images + 1) / 2  
    if save_images_separately:
        for i in range(n_images):
            img_path = os.path.join(save_path, f"generated_image_{i}.png")
            utils.save_image(generated_images[i], img_path, normalize=True)
            print(f"Generated image saved to {img_path}")
    else:
        output_path = os.path.join(save_path, "generated_from_checkpoint.png")
        utils.save_image(generated_images, output_path, nrow=4, normalize=True)
        print(f"Generated images saved to {output_path}")
