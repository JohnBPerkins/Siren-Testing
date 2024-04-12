import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt

plot_dir = "/home/wchung25/Siren-Testing"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def psnr(pred, target, max_pixel=1.0):
    mse = F.mse_loss(pred, target)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
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
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        return self.net(coords)

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
    
    # Store initial norm
    initial_weights_norm = torch.norm(torch.cat([p.data.flatten() for p in model.parameters()]))
    
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    losses = []
    psnr_values = []
    
    for step in range(total_steps):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
            model_output = model(model_input)
            loss = F.mse_loss(model_output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 steps
        if step % 10 == 0:  
                current_psnr = psnr(model_output, ground_truth)
                losses.append(loss.item())
                psnr_values.append(current_psnr.item())
                print(f"Step {step}: Loss {loss.item()}, PSNR: {current_psnr.item()}")

    # For the comparison image
    model_output = model(model_input).detach().cpu().view(28, 28)
    ground_truth = ground_truth.detach().cpu().view(28, 28)  # Reshape ground truth to image dimensions

    # Normalize the output for display
    model_output_norm = (model_output - model_output.min()) / (model_output.max() - model_output.min())
    ground_truth_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())

    # Create a figure with subplots
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(model_output_norm, cmap='gray')
    plt.title('Regenerated Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth_norm, cmap='gray')
    plt.title('Ground Truth Image')
    plt.axis('off')

    # Save the comparison image
    output_path = os.path.join(plot_dir, f'comparison_image_seed_{seed}.png')
    plt.savefig(output_path)
    plt.close()
    
    # Store final norm
    final_weights_norm = torch.norm(torch.cat([p.data.flatten() for p in model.parameters()]))
    
    return initial_weights_norm, final_weights_norm, losses, psnr_values

# Run the training with multiple seeds
results = {}
for seed in range(10):  # 10 different initializations
    results[seed] = train_model_with_seed(seed)

# Plotting and saving results for each seed
for seed, (initial_norm, final_norm, losses, psnr_values) in results.items():
    print(f"Seed {seed}: Initial Norm = {initial_norm}, Final Norm = {final_norm}")
    
    iterations = [10 * i for i in range(len(losses))]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, label='Loss')
    plt.title(f'Loss over Iterations for Seed {seed}')
    plt.xlabel('Iteration (every 10th)')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(plot_dir, f"loss_plot_seed_{seed}.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.subplot(1, 2, 1)
    plt.plot(iterations, psnr_values, label='PSNR', color='orange')
    plt.title(f'PSNR over Iterations for Seed {seed}')
    plt.xlabel('Iteration (every 10th)')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    psnr_plot_path = os.path.join(plot_dir, f"psnr_plot_seed_{seed}.png")
    plt.savefig(psnr_plot_path)
    plt.close()

    print(f"Plots for seed {seed} saved: {loss_plot_path} and {psnr_plot_path}")