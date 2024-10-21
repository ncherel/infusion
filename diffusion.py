import torch
import numpy as np


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 1000
        betas = np.linspace(1e-4, 0.02, self.T, dtype=np.float32)
        alphas = 1 - betas
        alphas_bar = np.cumprod(alphas)

        self.register_buffer("betas", torch.from_numpy(betas))
        self.register_buffer("alphas",torch.from_numpy(alphas))
        self.register_buffer("alphas_bar", torch.from_numpy(alphas_bar))


    def forward(self, x, T):
        noise = torch.randn_like(x)
        noisy = x * torch.sqrt(self.alphas_bar[T])[:,None,None,None,None] + torch.sqrt(1 - self.alphas_bar[T])[:,None,None,None,None] * noise
        return noisy
