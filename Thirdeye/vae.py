import os
os.environ['PYTHONPATH'] = '/home/avagnes/Gymnasium/Gymnasium'

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import get_threshold, Saliency


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=96*96, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Define loss function
    def vae_loss(self, reconstructed_x, x, mu, logvar):
        MSE = nn.functional.mse_loss(
            reconstructed_x, x.view(-1, self.input_dim), reduction='mean')
        # 可选：缩放MSE以匹配Keras版本中的做法
        MSE *= (96 * 96)  # 假设图像尺寸是96x96
        
        # KL散度部分保持不变
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


def train():
    # Hyperparameters
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    DEVICE = 'cpu'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    class ImageDataset(Dataset):
        def __init__(self):
            super().__init__()
            data = np.load('pictures.npy')
            model = PPO.load("./best_model.zip", device=DEVICE)
            saliency = Saliency(model.policy)
            def score_when_decrease(output):
                return -1.0 * output[0][0][0]
            data = torch.tensor(data[:10000], dtype=torch.float32).permute(0, 3, 1, 2).to(DEVICE)
            self.saliency_map = saliency(score_when_decrease, data, smooth_noise=0.2, smooth_samples=20)
        def __len__(self):
            return len(self.saliency_map)
        def __getitem__(self, index):
            return self.saliency_map[index], 0
    dataset = ImageDataset()
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    # Initialize model and optimizer
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()/len(data):.2f}')
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.2f}')
        torch.save(model.state_dict(), 'vae.pth')


def generate_data(number=10000):
    env = gym.make("CarRacing-v3",
                render_mode="rgb_array",)
    env = GrayscaleObservation(env, keep_dim=True)
    model = PPO.load("./best_model.zip")

    data = []
    terminated, truncated = True, True
    pbar = tqdm.tqdm(total=number)
    while len(data) < number:
        if terminated or truncated:
            obs, _ = env.reset()
            cnt = 0
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cnt += 1
        if cnt > 50:
            data.append(obs)
            pbar.update(1)
            # plt.imshow(obs)
            # plt.savefig('pic.jpg')
    np.save('pictures.npy', data)


if __name__ == '__main__':
    # generate_data()
    train()