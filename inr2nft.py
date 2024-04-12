import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from perceiver_pytorch import Perceiver
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from tqdm import tqdm

def psnr(pred, target, max_pixel=1.0):
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='xy'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.final_layer = nn.Linear(hidden_features, out_features) if outermost_linear else SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, return_features=False):
        coords = coords.clone().detach().requires_grad_(True)
        features = self.net(coords)
        output = self.final_layer(features)
        if return_features:
            return output, features
        return output
    
class NFT(nn.Module):
    def __init__(self, input_channels, latent_dim, depth=1, num_latents=64, attn_dropout=0.1, ff_dropout=0.1):
        super(NFT, self).__init__()
        self.perceiver = Perceiver(
            input_channels=input_channels,  # Adjusted to match your Siren's output feature dimensions
            input_axis=1,  # Number of axes in input feature, here it is just the feature dimension
            num_latents=num_latents,  # Number of latent vectors
            latent_dim=latent_dim,  # Dimension of each latent vector
            depth=depth,  # Number of cross-attention layers
            latent_heads=8,  # Number of heads in latent self-attention
            cross_heads=1,  # Number of heads in cross-attention
            cross_dim_head=50,  # Dimension per cross-attention head
            latent_dim_head=50,  # Dimension per self-attention head
            attn_dropout=attn_dropout,  # Dropout for attention
            ff_dropout=ff_dropout,  # Dropout for feed-forward layers
            weight_tie_layers=False,  # Whether to share weights across layers
            fourier_encode_data=False  # Whether to use Fourier encoding
        )
    
    def forward(self, x):
        # x is the input features from the Siren model
        return self.perceiver(x)



# Load MNIST Data
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
first_image = mnist_data[0][0]

# Prepare dataset
class MNISTImageFitting(Dataset):
    def __init__(self, image):
        super().__init__()
        self.coords = get_mgrid(28, 2)
        self.pixels = image.view(-1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

# Initialize dataset and dataloader
mnist_image_dataset = MNISTImageFitting(first_image)
dataloader = DataLoader(mnist_image_dataset, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
img_siren.cuda()

# Define seed setting and train for different initializations
def train_model_with_seed(seed, total_steps=350):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize the model
    model = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).cuda()
    
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    final_features = None
    
    for step in tqdm(range(total_steps), desc=f'Training Seed {seed}'):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            
            # Last step
            if step == total_steps - 1:  
                model_output, features = model(model_input, return_features=True)
                final_features = features.detach()  # Save features
            else:
                model_output = model(model_input)
            loss = F.mse_loss(model_output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return final_features

# Run the training with multiple seeds
results = {}
for seed in tqdm(range(10), desc='Processing Seeds'):  # 10 different initializations
    results[seed] = train_model_with_seed(seed)
    
# Assuming the features are extracted and the NFT is defined and instantiated
nft = NFT(input_channels=256, latent_dim=256)  # Adjust the input_channels to match your feature size
nft.cuda()

# Process each seed's output through NFT and measure norms
norms = {}
for seed, features in results.items():
    print("Features shape:", features.shape)
    transformed_features = nft(features.cuda())
    print("Transformed features shape:", transformed_features.shape)
    norms[seed] = torch.norm(transformed_features, dim=1).detach().cpu().numpy()

# Print or process norms
for seed, norm_values in norms.items():
    print(f"Norms of the latent vectors for seed {seed}:", norm_values)

