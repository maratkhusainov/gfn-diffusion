import numpy as np
import torch
from .base_set import BaseSet
from energies.mnist_2d_experiments.models import GeneratorMNISTWGAN, DiscriminatorMNISTWGAN

import sys
sys.path.append('..')


class WGANdens(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        self.device = device
        self.data_ndim = dim
        self.gen_mnist = GeneratorMNISTWGAN(2)
        self.gen_mnist.to(self.device)

        self.discr_mnist = DiscriminatorMNISTWGAN()
        self.discr_mnist.to(self.device)

        self.prior_mnist = torch.distributions.MultivariateNormal(
            torch.zeros(2).to(self.device), torch.eye(2).to(self.device))

        self.gen_mnist.load_state_dict(torch.load(
            '/content/gfn-diffusion/energy_sampling/energies/mnist_2d_experiments/weights/wgan_2d.ckpt')[0])
        self.discr_mnist.load_state_dict(torch.load(
            '/content/gfn-diffusion/energy_sampling/energies/mnist_2d_experiments/weights/wgan_2d.ckpt')[1])
        self.gen_mnist.eval()
        self.discr_mnist.eval()

    def energy(self, z):
        return (-self.discr_mnist(self.gen_mnist(z)).reshape(z.shape[0]) - self.prior_mnist.log_prob(z))
