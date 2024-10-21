import os
import argparse

from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusion import Diffusion
from model import Model
from data import SingleVideoTestDataset

parser = argparse.ArgumentParser(
                    prog = 'infuTest',
                    description = 'Test a model')
parser.add_argument("--video", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--interval-size", type=int, default=50)
parser.add_argument("--output", default="output")
parser.add_argument("--small", action="store_true", help="if enabled, resize the video to 256x256, otherwise 432x240")

args = parser.parse_args()

ckpt_folder = os.path.dirname(args.checkpoint)

device = "cuda"

# Create diffusion parameters and create model (weights will be loaded later)
diffusion = Diffusion().to(device)

model = Model()
model = model.to(device)
model.eval()

interval_size = args.interval_size

# Get video and mask as Tensors
dataset = SingleVideoTestDataset(args.video, args.mask, small=args.small)
dataloader = DataLoader(dataset, batch_size=1)
test_images, masks = next(iter(dataloader))

# Rest of the code expect mask to have one dimension only
masks = masks[:,:1]
masks = 1.0 * (masks > 0.0)

test_images = test_images.to(device)
masks = masks.to(device)
cmasks = masks.repeat(1, 3, 1, 1, 1)

# Mask the images
x = test_images * (1 - cmasks)
y = torch.randn_like(test_images)

for step in range(int(diffusion.T / interval_size)):
    model.load_state_dict(torch.load(join(ckpt_folder, f"model_{step:04d}.pth")))
    model = model.to(device)

    with torch.no_grad():
        t_end = diffusion.T - 1 - (step+1) * interval_size
        t_end = max(0, t_end)

        for t in range(diffusion.T-1 - step * interval_size, t_end, -1):
            t = torch.tensor([t])
            y0 = model(x, y, masks, t)

            # Clip
            y0 = torch.clip(y0, -1.0, 1.0)

            # Reproject y0
            y0 = masks * y0 + (1 - masks) * test_images

            # Mean and variance
            mean = diffusion.betas[t] * torch.sqrt(diffusion.alphas_bar[t-1]) / (1 - diffusion.alphas_bar[t]) * y0 + (1 - diffusion.alphas_bar[t-1]) * torch.sqrt(diffusion.alphas[t]) / (1 - diffusion.alphas_bar[t]) * y
            var = diffusion.betas[t] * (1 - diffusion.alphas_bar[t-1]) / (1 - diffusion.alphas_bar[t])

            # Sample from gaussian
            y = mean + torch.sqrt(var) * torch.randn_like(y)

os.makedirs(args.output, exist_ok=True)
for i in range(masks.shape[2]):
    save_image((1+y0[:,:,i])/2, join(args.output, f"pred_{i:04d}.png"), range=(-1,1))
