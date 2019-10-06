from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
from torch.autograd import Variable



class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):        

        h1 = F.relu(self.fc1(torch.cat((x, y), 1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z, y):
        #y = y.type(torch.cuda.FloatTensor if cuda_available else torch.FloatTensor).unsqueeze(1)
        #y = y.type(torch.cuda.FloatTensor if cuda_available else torch.FloatTensor)

        h3 = F.relu(self.fc3(torch.cat((z, y), 1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784), y)
        
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    
    BCE = F.binary_cross_entropy(recon_x, x, reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n and idx.dim()<=2
    idx2dim = idx.view(-1,1) # change from 1-dim tensor to 2-dim tensor
    onehot = torch.zeros(idx2dim.size(0),n).scatter_(1,idx2dim,1)

    return onehot

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(train_loader):
     
        data = data.round().to(device) # Binarize the data to test C-VAE on binary dataset
        label = idx2onehot(label,10).to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        
        loss = loss_function(recon_batch, data.view(-1, data.shape[2]*data.shape[3]), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    
    model.eval()
    
    test_loss = 0

    with torch.no_grad():
        
        for i, (data, label) in enumerate(test_loader):
            data = data.round().to(device)
            label = idx2onehot(label,10).to(device)

            recon_batch, mu, logvar = model(data, label)                                    
            test_loss += loss_function(recon_batch, data.view(-1, data.shape[2]*data.shape[3]), mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


no_cuda = False
cuda_available = not no_cuda and torch.cuda.is_available()

BATCH_SIZE =16
EPOCH = 100
SEED = 8

torch.manual_seed(SEED)

device = torch.device("cuda" if cuda_available else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_available else {}

train_loader = torch.utils.data.DataLoader( datasets.MNIST('./MNIST_data', train=True, download=True, transform=transforms.ToTensor()),batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./MNIST_data', train=False, transform=transforms.ToTensor()),batch_size=BATCH_SIZE, shuffle=True, **kwargs)

model = CVAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

image_save_path = 'images'

try:
    os.mkdir(image_save_path)
except OSError:
    print ("Creation of the directory %s failed" % image_save_path)
else:
    print ("Successfully created the directory %s " % image_save_path)


for epoch in range(1, EPOCH + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        
        # sampling
        sample = torch.randn(16, 20).to(device)
      
        c = torch.zeros(sample.shape[0],1).fill_(5).type(torch.LongTensor)
      
        c = idx2onehot(c,10).to(device)

        sample = model.decode(sample, c).cpu()

        generated_image = sample.round()
        
        save_image(generated_image.view(16, 1, 28, 28),os.path.join(image_save_path,'sample_{}.png'.format(str(epoch))))


