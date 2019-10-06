"""
BSD 3-Clause License

Copyright (c) 2017, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from torch import nn, optim
from torch.nn import functional as F
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE_Vanilla(nn.Module):

    def __init__(self,input_size = 784, mid_layer_size = 400, latent_size = 30,output_size=784):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,mid_layer_size)  #input observed data
        self.mu = nn.Linear(mid_layer_size,latent_size)  # mean
        self.logvar = nn.Linear(mid_layer_size, latent_size) # variance
        self.fc2 = nn.Linear(latent_size, mid_layer_size) # sampled latent variables
        self.fc3 = nn.Linear(mid_layer_size,output_size) # reconstruction

    
    def encoder(self,X):
        o1 = F.relu(self.fc1(X))
        return self.mu(o1), self.logvar(o1)   # approximate the mean and logarisim of variance

    def reparameterize(self, mu,logvar):
        
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decoder(self,sampled_latent_z):
        o2 = F.relu(self.fc2(sampled_latent_z))
        return torch.sigmoid(self.fc3(o2))

    def forward(self, X):
        mu, logvar = self.encoder(X.view(-1,self.input_size))
        sampled_latent_z = self.reparameterize(mu,logvar)
        return self.decoder(sampled_latent_z), mu, logvar
    
    def loss_function(self, input_X, output_X, mu,logvar):

        reconstruction_loss = - F.binary_cross_entropy(output_X, input_X.view(-1,self.input_size), reduction = 'sum')
        
        KL_divergence = 0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
        
        return - (reconstruction_loss + KL_divergence) # we want to maximize "reconstruction_loss - KL_divergence"





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE_Vanilla().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
BATCH_SIZE= 32
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),batch_size=BATCH_SIZE, shuffle=True)


def VAE_Vanilla_train():
    model.train() # this is for dropout 
    train_loss = 0

    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_data, mu, logvar  = model(data)
        loss = model.loss_function(data,recon_data,mu,logvar)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss / len(train_loader.dataset)
    
def VAE_Vanilla_test():
    model.eval() # this is for dropout removal
    test_loss = 0
    with torch.no_grad(): # gradient calculation is not needed
        for data, _ in test_loader:
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            test_loss += model.loss_function(data,recon_data,mu,logvar)

    return test_loss/ len(test_loader.dataset)


if __name__ == "__main__":
    for epoch in range(1, 20):
        average_train_loss_per_epoch = VAE_Vanilla_train()
        print('====> Epoch: {} Average Training loss: {:.4f}'.format(epoch, average_train_loss_per_epoch))

        average_test_loss_per_epoch = VAE_Vanilla_test()
        print('====> The corresponding average Test loss: {:.4f}'.format(average_test_loss_per_epoch))

        with torch.no_grad():
            sample = torch.randn(4, 30).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(4, 1, 28, 28),'./results/sample_' + str(epoch) + '.png')