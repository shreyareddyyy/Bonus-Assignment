import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
lr = 0.0002
latent_dim = 100
num_classes = 10
embedding_dim = 50
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)
epochs = 30
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# DataLoader for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root=".", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(embedding_dim + int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)
        d_input = torch.cat((img_flat, label_embedding), -1)
        validity = self.model(d_input)
        return validity

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers and loss
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Fixed noise for visualization
fixed_noise = torch.randn(num_classes, latent_dim, device=device)
fixed_labels = torch.arange(num_classes, device=device)

# Training loop
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size_curr = imgs.size(0)

        real = torch.ones(batch_size_curr, 1, device=device)
        fake = torch.zeros(batch_size_curr, 1, device=device)

        imgs, labels = imgs.to(device), labels.to(device)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)

        generated_imgs = generator(z, gen_labels)
        validity = discriminator(generated_imgs, gen_labels)

        g_loss = criterion(validity, real)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        real_validity = discriminator(imgs, labels)
        d_real_loss = criterion(real_validity, real)

        # Fake images
        fake_validity = discriminator(generated_imgs.detach(), gen_labels)
        d_fake_loss = criterion(fake_validity, fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        if i % 200 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Save generated samples for visualization
    generator.eval()
    with torch.no_grad():
        sample_imgs = generator(fixed_noise, fixed_labels)
        sample_imgs = (sample_imgs + 1) / 2  # Rescale to [0, 1]
        save_image(make_grid(sample_imgs, nrow=num_classes), f"{output_dir}/epoch_{epoch:03}.png")
    generator.train()

print("Training Complete. Generated images saved in:", output_dir)
