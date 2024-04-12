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
import matplotlib.pyplot as plt
from tqdm import tqdm

plot_dir = "/home/wchung25/Siren-Testing"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def psnr(pred, target, max_pixel=1.0):
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def get_mgrid(sidelen, dim=2):
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
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.net = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)] + \
                   [SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0) for _ in range(hidden_layers)]
        self.final_layer = nn.Linear(hidden_features, out_features) if outermost_linear else SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords, return_features=False):
        features = self.net(coords)
        output = self.final_layer(features)
        return (output, features) if return_features else output

class NFT(nn.Module):
    def __init__(self, input_channels, latent_dim, depth=1, num_latents=64, attn_dropout=0.1, ff_dropout=0.1):
        super(NFT, self).__init__()
        self.perceiver = Perceiver(input_channels=input_channels, input_axis=1, num_latents=num_latents, latent_dim=latent_dim, depth=depth, latent_heads=8, cross_heads=1, cross_dim_head=50, latent_dim_head=50, attn_dropout=attn_dropout, ff_dropout=ff_dropout, weight_tie_layers=False, fourier_encode_data=False)

    def forward(self, x):
        return self.perceiver(x)

# Load MNIST Data
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
first_image = mnist_data[0][0]

# Prepare dataset
class MNISTImageFitting(Dataset):
    def __init__(self, image):
        self.coords = get_mgrid(28, 2)
        self.pixels = image.view(-1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.pixels

# Initialize dataset and dataloader
mnist_image_dataset = MNISTImageFitting(first_image)
dataloader = DataLoader(mnist_image_dataset, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).cuda()
nft = NFT(input_channels=256, latent_dim=256).cuda()

def train_and_transform(seed, total_steps=350):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.Adam(img_siren.parameters(), lr=1e-4)
    initial_norm = torch.norm(torch.cat([p.data.flatten() for p in img_siren.parameters()]))
    losses = []
    psnr_values = []

    for step in tqdm(range(total_steps), desc=f'Training Seed {seed}'):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            model_output, features = img_siren(model_input, return_features=True)
            loss = F.mse_loss(model_output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                current_psnr = psnr(model_output, ground_truth)
                losses.append(loss.item())
                psnr_values.append(current_psnr.item())

    final_norm = torch.norm(torch.cat([p.data.flatten() for p in img_siren.parameters()]))
    transformed_features = nft(features.cuda())
    norms = torch.norm(transformed_features, dim=1).detach().cpu().numpy()

    return initial_norm.item(), final_norm.item(), losses, psnr_values, norms

results = {}
for seed in range(10):
    results[seed] = train_and_transform(seed)

for seed, (initial_norm, final_norm, losses, psnr_values, norms) in results.items():
    print(f"Seed {seed}: Initial Norm = {initial_norm}, Final Norm = {final_norm}, NFT Norms = {norms}")
