import random

from PIL import Image

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

import torchvision
from torchvision.transforms import Normalize, Compose, Resize

import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


class SingleMaskedVideoTrainDataset(IterableDataset):
    def __init__(self, image_path, mask_path, small=False, window=12):
        super().__init__()
        self.video, _, _ = torchvision.io.read_video(image_path)
        self.mask, _, _ = torchvision.io.read_video(mask_path)
        size = (256,256) if small else (240, 432)
        self.transforms = Compose([
            Resize(size, antialias=True),
            Normalize(0.5, 0.5)  # Change range [0,1] -> [-1, 1]
        ])
        self.mask_transforms = Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.window = window

    def __iter__(self):
        def next_crop():
            T = self.video.shape[0]

            # Change window training size
            hw = self.window // 2

            # Use all frames or split train / test temporally
            T_end = T

            while True:
                # Draw a time for a frame in the video (except the center of the video)
                t = np.random.randint(hw, T_end - hw + 1)
                frames = [self.transforms(self.video[i].permute(2, 0, 1) / 255.0) for i in range(t - hw, t + hw, 1)]
                masks = [self.mask_transforms(self.mask[i].permute(2, 0, 1) / 255.0) for i in range(t - hw, t + hw, 1)]
                yield torch.stack(frames, dim=1), torch.stack(masks, dim=1)

        return iter(next_crop())


class SingleVideoTestDataset(Dataset):
    def __init__(self, image_path, mask_path, small=False, n_frames=1000):
        """Test on video, n_frames can be used to process only the first n frames of a video if running into
        memory issues
        """
        super().__init__()
        self.video, _, _ = torchvision.io.read_video(image_path)
        self.mask, _, _ = torchvision.io.read_video(mask_path)
        size = (256,256) if small else (240, 432)
        self.transforms = Compose([
            Resize(size, antialias=True),
            Normalize(0.5, 0.5)  # Change range [0,1] -> [-1, 1]
        ])
        self.mask_transforms = Resize(size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.n_frames = n_frames

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        T = self.video.shape[0]
        frames = [self.transforms(self.video[i].permute(2, 0, 1) / 255.0) for i in range(min(self.n_frames, T))]
        masks = [self.mask_transforms(self.mask[i].permute(2, 0, 1) / 255.0) for i in range(min(self.n_frames, T))]
        return torch.stack(frames, dim=1), torch.stack(masks, dim=1)


"""
Copyright JingjingRenabc
Code for generating moving shapes in videos for synthetic masks
"""

def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve.
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle,
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    try:
        # Old matplotlib version
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    except:
        # recent matplotlib
        data = np.asarray(fig.canvas.renderer.buffer_rgba())[:,:,:3]
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region



def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # get a random shape
    height = random.randint(imageHeight//2, imageHeight)
    width = random.randint(imageWidth//2, imageWidth)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=2)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks*video_length
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(2, 0.5), maxInitSpeed=2)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
    return masks



def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X, Y, imageHeight, imageWidth, lineVelocity, region_size, maxLineAcceleration=(3, 0.5), maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(
        lineVelocity, maxLineAcceleration, dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0) or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity

if __name__ == '__main__':
    """Test the mask generation algorithm"""
    vid = create_random_shape_with_random_motion(48)
    for i in range(len(vid)):
        vid[i].save(f"{i:03d}.png")
