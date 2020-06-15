import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.define_layers()

    def define_layers(self):
        self.linear01 = nn.Linear(self.latent_dim, 128)
        self.leaky_relu = nn.LeakyReLU()
        self.linear02 = nn.Linear(128, 128)
        # self.leaky_relu
        self.linear03 = nn.Linear(128, 128)
        # self.leaky_relu
        self.output_layer = nn.Linear(128, self.output_dim)

    def forward(self, input_tensor):
        intermediate = self.linear01(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear02(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear03(intermediate)
        intermediate = self.leaky_relu(intermediate)
        output_tensor = self.output_layer(intermediate)
        return output_tensor


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.define_layers()

    def define_layers(self):
        self.linear01 = nn.Linear(self.input_dim, 128)
        self.leaky_relu = nn.LeakyReLU()
        self.linear02 = nn.Linear(128, 128)
        # self.leaky_relu
        self.linear03 = nn.Linear(128, 128)
        # self.leaky_relu
        self.output_layer = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        intermediate = self.linear01(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear02(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear03(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.output_layer(intermediate)
        output_tensor = self.sigmoid(intermediate)
        return output_tensor


class VanillaGAN():
    def __init__(self, generator, discriminator, noise_fn, data_fn, batch_size=32):
        self.generator = generator
        self.discriminator = discriminator
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size

    def generate_samples(latent_vec=None, num=None):
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        


gen = Generator(1, 1)
disc = Discriminator(1)
X = torch.linspace(-1, 1, 200)
with torch.no_grad():
    Y = gen(X.unsqueeze(1))
    C = disc(Y)
plt.scatter(X, Y.squeeze(), label="Gen")
plt.scatter(X, C.squeeze(), label="Disc")
plt.legend()
plt.show()
