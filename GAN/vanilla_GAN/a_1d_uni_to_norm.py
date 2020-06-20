import os

import matplotlib.pyplot as plt
import torch

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # GPU speedup not worth it here
DIR, FILE = os.path.split(os.path.abspath(__file__))
NAME = '.'.join(FILE.split('.')[:-1])
OUTPUT_DIR = os.path.join("images", NAME)

def noise_fn(num):
    return torch.rand((num, LATENT_DIM), device=DEVICE)

def target_fn(num):
    return torch.randn((num, 1), device=DEVICE)


os.makedirs()

gen = Generator(LATENT_DIM, [32, 32, 32, 1], output_activation=None)
disc = Discriminator(1, [32, 32, 32, 1])
gan = VanillaGAN(gen, disc, noise_fn, target_fn,)
real_samples = np.random.normal(0, 1, 1000)
real_samples = np.sort(real_samples)
for i in range(1000):
    print(i)
    for _ in range(100):
        gan.train_step()
    generated_samples = gan.generate_samples(num=1000).cpu().numpy().flatten()
    generated_samples = np.sort(generated_samples)
    plt.cla()
    plt.scatter(real_samples, np.linspace(0, 1, len(real_samples)), c='k', s=2)
    plt.scatter(generated_samples, np.linspace(0, 1, len(generated_samples)), c='r', s=2)
    plt.draw()
    plt.pause(0.001)
input()
