Application of VAEs with examples and implementations where relevant:

### 1. Image Generation

```python
import torch
import torch.nn as nn

class ImageVAE(nn.Module):
    def __init__(self, image_channels=3):
        super(ImageVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * 4 * 4, 256)
        self.fc_var = nn.Linear(128 * 4 * 4, 256)

        # Decoder
        self.decoder_input = nn.Linear(256, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
```

### 2. Data Compression

```python
class CompressionVAE(nn.Module):
    def __init__(self, input_dim, compressed_dim):
        super(CompressionVAE, self).__init__()

        # Encoder (compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Latent space (compressed representation)
        self.fc_mu = nn.Linear(256, compressed_dim)
        self.fc_var = nn.Linear(256, compressed_dim)

        # Decoder (decompression)
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
```

### 3. Anomaly Detection

```python
class AnomalyVAE:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def detect_anomaly(self, x):
        # Reconstruct the input
        x_recon, mu, logvar = self.model(x)

        # Compute reconstruction error
        recon_error = torch.mean((x - x_recon) ** 2, dim=1)

        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Total anomaly score
        anomaly_score = recon_error + kl_div

        # Detect anomalies
        is_anomaly = anomaly_score > self.threshold
        return is_anomaly, anomaly_score

# Usage example
def detect_anomalies(data_loader, anomaly_detector):
    anomalies = []
    for batch in data_loader:
        is_anomaly, scores = anomaly_detector.detect_anomaly(batch)
        anomalies.extend(is_anomaly.tolist())
    return anomalies
```

### 4. Drug Discovery

```python
class MoleculeVAE(nn.Module):
    def __init__(self, num_features):
        super(MoleculeVAE, self).__init__()

        # Molecular feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(256, 128)
        self.fc_var = nn.Linear(256, 128)

        # Molecular feature decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_features)
        )

    def generate_molecules(self, num_samples):
        # Sample from latent space
        z = torch.randn(num_samples, 128)
        # Generate molecular features
        return self.decoder(z)
```

### 5. Style Transfer

```python
class StyleTransferVAE(nn.Module):
    def __init__(self):
        super(StyleTransferVAE, self).__init__()

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def transfer_style(self, content_img, style_img):
        content_features = self.content_encoder(content_img)
        style_features = self.style_encoder(style_img)
        combined_features = torch.cat([content_features, style_features], dim=1)
        return self.decoder(combined_features)
```

### 6. Data Augmentation

```python
class AugmentationVAE:
    def __init__(self, model):
        self.model = model

    def generate_augmented_samples(self, x, num_samples=5):
        # Encode input to latent space
        mu, logvar = self.model.encode(x)

        augmented_samples = []
        for _ in range(num_samples):
            # Sample from latent space with small perturbations
            z = self.model.reparameterize(mu, logvar)
            z_perturbed = z + 0.1 * torch.randn_like(z)

            # Generate new sample
            augmented = self.model.decode(z_perturbed)
            augmented_samples.append(augmented)

        return torch.stack(augmented_samples)

# Usage example
def augment_dataset(dataset, aug_vae, samples_per_image=5):
    augmented_data = []
    for x in dataset:
        augmented = aug_vae.generate_augmented_samples(x, samples_per_image)
        augmented_data.extend(augmented)
    return augmented_data
```

Each of these applications demonstrates different aspects of VAEs:

1. **Image Generation** focuses on creating new, realistic images
2. **Data Compression** reduces data dimensionality while preserving important features
3. **Anomaly Detection** identifies unusual patterns by comparing reconstruction error
4. **Drug Discovery** helps generate new molecular structures
5. **Style Transfer** combines content and style information
6. **Data Augmentation** creates variations of existing data

These implementations can be further customized based on specific requirements and datasets. Would you like to explore any particular application in more detail?
