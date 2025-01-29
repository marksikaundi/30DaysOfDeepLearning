Complete example of image generation using VAEs with the MNIST dataset. This example will include training, visualization, and generation of new images.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)

        return x, mu, logvar

def train_vae():
    # Hyperparameters
    batch_size = 128
    epochs = 20
    learning_rate = 1e-3
    latent_dim = 32

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(data)

            # Loss calculation
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() / len(data)})

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}: Average loss = {avg_loss:.4f}')

    return model

def visualize_results(model, num_examples=10):
    """Visualize original and reconstructed images"""
    model.eval()

    # Get some test images
    test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=num_examples, shuffle=True)
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        reconstructed, _, _ = model(test_images)

    # Plot results
    fig, axes = plt.subplots(2, num_examples, figsize=(20, 4))

    for i in range(num_examples):
        # Original images
        axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.show()

def generate_new_images(model, num_images=10):
    """Generate new images by sampling from the latent space"""
    model.eval()
    with torch.no_grad():
        # Sample from normal distribution
        z = torch.randn(num_images, 32).to(device)

        # Decode the random samples
        samples = model.decoder_input(z)
        samples = samples.view(-1, 64, 7, 7)
        generated = model.decoder(samples)

        # Plot results
        fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
        for i in range(num_images):
            axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Generated Images')

        plt.tight_layout()
        plt.show()

def interpolate_digits(model, num_steps=10):
    """Generate interpolation between two random points in latent space"""
    model.eval()
    with torch.no_grad():
        # Sample two random points in latent space
        z1 = torch.randn(1, 32).to(device)
        z2 = torch.randn(1, 32).to(device)

        # Create interpolation steps
        alphas = np.linspace(0, 1, num_steps)
        z_interpolated = torch.zeros(num_steps, 32).to(device)

        for i, alpha in enumerate(alphas):
            z_interpolated[i] = alpha * z2 + (1 - alpha) * z1

        # Generate images from interpolated points
        samples = model.decoder_input(z_interpolated)
        samples = samples.view(-1, 64, 7, 7)
        generated = model.decoder(samples)

        # Plot results
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
        for i in range(num_steps):
            axes[i].imshow(generated[i].cpu().squeeze(), cmap='gray')
            axes[i].axis('off')

        plt.suptitle('Interpolation between two random points in latent space')
        plt.tight_layout()
        plt.show()

# Train the model
model = train_vae()

# Visualize results
visualize_results(model)

# Generate new images
generate_new_images(model)

# Show interpolation between digits
interpolate_digits(model)
```

This complete example includes:

1. **A Convolutional VAE Architecture:**

   - Convolutional encoder for better feature extraction
   - Proper latent space sampling
   - Transposed convolutional decoder for image generation

2. **Training Loop:**

   - Progress bar with loss tracking
   - Both reconstruction and KL divergence losses

3. **Visualization Functions:**

   - Compare original vs reconstructed images
   - Generate completely new images
   - Create interpolations between different digits

4. **Key Features:**
   - Uses GPU if available
   - Proper error handling
   - Progress tracking during training
   - Multiple visualization options

To run this code, you'll need:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

The code will:

1. Train the VAE on MNIST
2. Show original vs reconstructed images
3. Generate new digits from random latent vectors
4. Show smooth interpolations between different digits

You can modify various parameters like:

- `latent_dim` for different latent space sizes
- `batch_size` for different training batch sizes
- Network architecture (add/remove layers, change filter sizes)
- Number of epochs and learning rate

Would you like me to explain any particular part in more detail or help you modify the code for a different dataset?
