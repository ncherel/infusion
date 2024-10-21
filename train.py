import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from torchvision.transforms import ToTensor

from data import SingleMaskedVideoTrainDataset, create_random_shape_with_random_motion
from diffusion import Diffusion
from model import Model

parser = argparse.ArgumentParser(
                    prog = 'InfuTrain',
                    description = 'Train a model on a single video')
parser.add_argument("--video", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--small", action="store_true", help="if enabled, resize the video to 256x256, otherwise 432x240")
parser.add_argument("--steps", type=int, default=100, help="number of training steps for each interval")
parser.add_argument("--interval-size", type=int, default=50, help="size of intervals, default is 50 for 1000 diffusion steps")

args = parser.parse_args()

writer = SummaryWriter(f"runs/{os.path.basename(args.video)}")

device = "cuda"
batch_size = 1

train_dataset = SingleMaskedVideoTrainDataset(args.video, args.mask, small=args.small)
dataloader = DataLoader(train_dataset, batch_size=batch_size)

diffusion = Diffusion().to(device)

model = Model()
model = model.to(device)

print(f"Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = Adam(model.parameters(), lr=1e-4)

interval_size = args.interval_size

tf = ToTensor()

for step in range(int(diffusion.T / interval_size)):
    T_min = diffusion.T - (step+1) * interval_size
    T_max = diffusion.T - step * interval_size

    for i, (images, test_masks) in zip(range(args.steps), dataloader):
        images = images.to(device)
        test_masks = test_masks.to(device)

        # Strict binary mask
        test_masks = 1.0 * (test_masks > 0.0)

        # Start by removing test content in images
        images = (1 - test_masks) * images

        # Create the training masks and convert to PyTorch
        masks = create_random_shape_with_random_motion(images.shape[2], imageHeight=images.shape[3], imageWidth=images.shape[4])
        masks = torch.stack([tf(m) for m in masks], dim=1).to(device).unsqueeze(0)
        masks = masks.repeat(images.shape[0], 1, 1, 1, 1)

        # Extend the "mask" information to the full mask (= OR operation)
        masks = 1 - (1 - masks) * (1 - test_masks)
        masks = masks[:,:1]

        cmasks = masks.repeat(1, 3, 1, 1, 1)

        # Mask the images
        x = images * (1 - cmasks)
        t = torch.randint(T_min, T_max, size=(images.shape[0],), device=device)
        y = diffusion.forward(images, t)

        optimizer.zero_grad()

        # Apply model on x, y, t
        denoised = model(x, y, masks, t)

        # Loss function is the product of the
        loss = F.mse_loss((1 - test_masks) * images, (1 - test_masks) * denoised)

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"{writer.log_dir}/model_{step:04d}.pth")

torch.save(model.state_dict(), f"{writer.log_dir}/model_last.pth")
writer.close()
