import os
import numpy as np

from torchvision.utils import save_image

from torch.autograd import Variable

import torch.nn as nn
import torch.autograd as autograd
import torch

import gc

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import GENERATED_IMAGES_FOLDER, CHECKPOINT_FOLDER, RESULTS_FOLDER, TRAINING_IMAGES_CHECKPOINT_FOLDER, INTERNAL_REPRESENTATIONS_FOLDER, CUDA as device, G_FEATURES_FILENAME

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
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
            return x.view(x.shape[0], *self.img_shape)
        else:
            return self.model(x).view(x.shape[0], *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()

        self.img_shape =  img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

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

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake.cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def save_generator_features(epoch, features, output_dir=INTERNAL_REPRESENTATIONS_FOLDER):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # np.save(os.path.join(output_dir, f'{G_FEATURES_FILENAME}_epoch_{epoch}.npy'), features)
    output_path = os.path.join(output_dir, f'{G_FEATURES_FILENAME}_epoch_{epoch}.npy')
    np.save(output_path, features)
    print(f"Saved generator features for epoch {epoch} to {output_path}")
    
    # Clear caches and force garbage collection after saving features
    del features
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Run garbage collection to free memory

def train(dataloader, n_epochs=500, save_interval=10, latent_dim=100, n_critic=5, lr=0.00005, b1=0.5, b2=0.999, selected_epochs=None):

    if selected_epochs is None:
        selected_epochs = [epoch for epoch in range(n_epochs) if (epoch % save_interval == 0) or (epoch < 10) or (epoch > n_epochs / 2 - 5 and epoch < n_epochs / 2 + 5) or (epoch > n_epochs - 10)]
        selected_epochs.sort()
    
    print(f"Selected epochs: {selected_epochs}")

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if device is not None:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if device is not None else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    # internal_representations = dict()

    batches_done = 0
    for epoch in range(n_epochs):
        batch_features = {}
        for batch_idx, (imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
                
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Sample noise as generator input z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            z = torch.randn(imgs.shape[0], latent_dim).cuda()            
            fake_imgs = generator(z, store_features=False)
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.cuda().data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if batch_idx % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                # fake_imgs = generator(z, store_features=epoch in selected_epochs) 
                
                if epoch in selected_epochs:# and batch_idx % 10 == 0:
                    # Store features for the current batch
                    fake_imgs = generator(z, store_features=True, epoch=epoch)
                    batch_features[batch_idx] = generator.features_dict
                else:
                    fake_imgs = generator(z, store_features=False, epoch=epoch)
                    
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                batches_done += n_critic
                
                # if epoch in selected_epochs:
                #     internal_representations[batch_idx] = generator.features_dict

        if epoch in selected_epochs:
            save_generator_features(epoch, batch_features)
            if fake_imgs.shape[0] > 16:
                fake_imgs = fake_imgs[:16]
            save_image(fake_imgs.data, os.path.join(TRAINING_IMAGES_CHECKPOINT_FOLDER, f"{epoch}.png"), nrow=4, normalize=True)
            save_model_checkpoints(generator=generator, discriminator=discriminator, epoch=epoch, last_epoch=(epoch==n_epochs-1))

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, d_loss.item(), g_loss.item()))

        # Clear caches and force garbage collection after each epoch
        torch.cuda.empty_cache()
        gc.collect()

def generate_images(n_images, split, latent_dim=100, checkpoint_path=None, save_path=GENERATED_IMAGES_FOLDER, map_location='cpu'):
    # Load the model
    generator = Generator()
    if map_location != 'cpu':
        generator = generator.cuda()
    generator.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    generator.eval()
    # If the destination folder does not exist, create it in one line
    os.makedirs(os.path.join(save_path, split), exist_ok=True)
    z = torch.randn(n_images, latent_dim, device=map_location)
    # Generate images
    gen_imgs = generator(z, store_features=False)
    for i in range(n_images):
        save_image(gen_imgs.data[i], os.path.join(save_path, split, f"{i}.png"), normalize=True)