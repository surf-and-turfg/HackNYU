from model import unet
from sampler import T
from sampler import IMG_SIZE
from sampler import BATCH_SIZE
import matplotlib as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scheduler import forward_diffusion_sample
from scheduler import linear_beta_schedule
from scheduler import get_index_from_list
from scheduler import betas
from scheduler import *
from dataloader import *
from dataloader import data_loader
import numpy as np
from model import model
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "model.pth"
model = unet() 

model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)

model.eval()
def sample_timestep(x, t):
    #kept getting fucking errors with the devices being wrong
    betas_t = get_index_from_list(betas, t, x.shape).to(device)  # Ensure betas are on the correct device
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape).to(device)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape).to(device)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    ).to(device)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape).to(device)  # Ensure posterior variance is on the correct device
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x).to(device)  # Ensure noise is on the same device
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()            

sample_plot_image()
