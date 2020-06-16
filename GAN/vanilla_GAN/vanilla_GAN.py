import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 1

class Generator(nn.Module):
    def __init__(self, latent_dim, layers, output_activation=None):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self._define_layers(layers)

    def _define_layers(self, layers):
        self.module_list = nn.ModuleList()
        last_layer = self.latent_dim
        for index, width in enumerate(layers):
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU())
        else:
            if self.output_activation is not None:
                self.module_list.append(self.output_activation())

    def forward(self, input_tensor):
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.define_layers()

    def define_layers(self):
        self.linear01 = nn.Linear(self.input_dim, 32)
        self.leaky_relu = nn.LeakyReLU()
        self.linear02 = nn.Linear(32, 32)
        # self.leaky_relu
        self.linear03 = nn.Linear(32, 32)
        # self.leaky_relu
        self.output_layer = nn.Linear(32, 1)
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
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(discriminator.parameters(), lr=2e-3)
        self.optim_g = optim.Adam(generator.parameters(), lr=4e-5)
        self.target_ones = torch.ones((batch_size, 1))
        self.target_zeros = torch.zeros((batch_size, 1))

    def generate_samples(self, latent_vec=None, num=None):
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()
        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()
        real_samples = self.data_fn(self.batch_size)
        with torch.no_grad():
            latent_vec = self.noise_fn(self.batch_size)
            generated = self.generator(latent_vec)

        classifications_real = self.discriminator(real_samples)
        loss_real = self.criterion(classifications_real, self.target_ones)
        classifications_generated = self.discriminator(generated)
        loss_generated = self.criterion(classifications_generated, self.target_zeros)
        loss = loss_real + loss_generated
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_generated.item()

    def train_step(self):
        loss_d = self.train_step_discriminator()
        loss_g = self.train_step_generator()
        return loss_g, loss_d


def noise_fn(num):
    return torch.rand((num, LATENT_DIM))


def target_fn(num):
    return torch.randn((num, 1))


plt.ion()
plt.figure()
plt.show()
gen = Generator(LATENT_DIM, [32, 32, 32, 1], output_activation=None)
disc = Discriminator(1)
gan = VanillaGAN(gen, disc, noise_fn, target_fn,)
real_samples = np.random.normal(0, 1, 1000)
real_samples = np.sort(real_samples)
for i in range(1000):
    print(i)
    for _ in range(100):
        gan.train_step()
    generated_samples = gan.generate_samples(num=1000).numpy().flatten()
    generated_samples = np.sort(generated_samples)
    plt.cla()
    plt.scatter(real_samples, np.linspace(0, 1, len(real_samples)), c='k', s=2)
    plt.scatter(generated_samples, np.linspace(0, 1, len(generated_samples)), c='r', s=2)
    plt.draw()
    plt.pause(0.001)
