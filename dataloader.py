import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scheduler import forward_diffusion_sample
from scheduler import T
import numpy as np
from PIL import Image
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np

data_dir = "./timri"

IMG_SIZE = 64
BATCH_SIZE = 32

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to a smaller size for visualization
    transforms.RandomHorizontalFlip(),  # Random flip for augmentation
    transforms.ToTensor(),  # Convert image to tensor and normalize [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1)  # Scale the tensor values between [-1, 1]
])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))



# Load dataset and dataloader
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
image = next(iter(data_loader))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

#just displays an image with the noise adjuster
#this is just to calibrate it
for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_tensor_image(img)
    
plt.show()
