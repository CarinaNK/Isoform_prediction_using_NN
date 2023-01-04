import pandas as pd
import torch.nn as nn
import torch.utils
import torch.distributions
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Dataset
class CSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples, seperator):
        self.path = path
        self.chunksize = chunksize
        self.seperator = seperator
        self.len = nb_samples // self.chunksize

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                sep =  self.seperator,
                skiprows = index * self.chunksize,
                chunksize = self.chunksize,
                index_col = 0))
        
        x = x.values
        x = np.float32(x)
        x = np.log10(x +1)
        x = torch.from_numpy(x)

        return x

    def __len__(self):
        return self.len


class Encoder(nn.Module):
    def __init__(self, latent_dims, middims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(34557, middims)
        self.linear2 = nn.Linear(middims, latent_dims)

    def forward(self, x):
        relu = nn.ReLU()
        x = relu(self.linear1(x))
        return self.linear2(x)     

class Decoder(nn.Module):
    def __init__(self, latent_dims, middims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, middims)
        self.linear2 = nn.Linear(middims, 34557)

    def forward(self, z):
        relu = nn.ReLU()
        z = relu(self.linear1(z))
        z = self.linear2(z)
       
        return z

class Autoencoder(nn.Module):
    def __init__(self, latent_dims, middims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, middims)
        self.decoder = Decoder(latent_dims, middims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z) , z
        # this the z we want for the nn


def train(autoencoder, data, epochs, weight_decay, learning_rate):
    opt = torch.optim.Adam(autoencoder.parameters(), weight_decay = weight_decay, lr = learning_rate)
    loss=nn.MSELoss()

    recon=[]

    for epoch in range(epochs):
    
        with open('output/tmp_100epochs.txt', 'w') as f:
            f.write(f'{epoch} out of {epochs} epochs\n')

        for  x in data:
            
            x = x.to(device) # GPU
            x = x[0].float()
            opt.zero_grad()
            x_hat,z = autoencoder(x)
            reconloss = loss(x,x_hat)
            reconloss.backward()
            opt.step()

       # if epoch % 100 == 0:
       #             
       #     reconloss_batches = []
       #     with torch.no_grad():
       #
       #         autoencoder.eval()
       #         
       #         for  x in data:
       #             x = x.to(device)
       #             x = x[0].float()
       #             x_hat,z = autoencoder(x)
       #             reconloss = loss(x,x_hat)
       #
       #             reconloss_batches.append(reconloss.item())
       #
       #         autoencoder.train()
       #         
       #     # Append average validation accuracy to list.
       #     recon.append(np.mean(reconloss_batches))

    return autoencoder , recon


def main(train_file, data_file, learning_rate, weight_decay, latent_dims, middims):
    
    dataset = CSVDataset(train_file, chunksize = 5000, nb_samples = 168668, seperator = '\t') # set sample size based on number of samples in dataset
    auto_data = DataLoader(dataset, batch_size = 1, num_workers = 8, shuffle=False)

    autoencoder = Autoencoder(latent_dims, middims).to(device) # GPU
    
    autoencoder, _ = train(autoencoder, auto_data, 100, weight_decay, learning_rate)

    gene_data = CSVDataset(data_file, chunksize = 5000, nb_samples = 17382, seperator = '\t') # set sample size based on number of samples in dataset
    gene_dataloader = DataLoader(gene_data, batch_size = 1, num_workers = 0, shuffle=False)

    transformed_data = pd.DataFrame()
    for x in gene_dataloader:
        x = x.to(device)
        x = x[0].float()

        _, tmp = autoencoder(x)
        tmp = tmp.cpu().detach().numpy()
        tmp = pd.DataFrame(tmp)
        transformed_data = pd.concat([transformed_data, tmp])

    return(transformed_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help = "File used to train the VAE, TSV/CSV format is expected")
    parser.add_argument('--data_file', help = "Data file to be reduced with autoincoder, TSV/CSV format is expected")

    parser.add_argument('-L', '--learning_rate', type = float, default = .001, required = False, help = "Learning rate for VAE, should be a number between 10^-6 and 1")
    parser.add_argument('-W' ,'--weight_decay', type = float, default = .0001, required = False, help = "Weight decay for VAE, should be a number between 0 and 0.1")
    parser.add_argument('-l', '--latent_dims', type = int, default = 1000, required = False, help = "Dimension of latent space")
    parser.add_argument('-m', '--mid_dims', type = int, default = 8000, required = False, help = "Dimension of the middel layer")


    args = parser.parse_args()

    print(main(args.train_file, args.data_file, args.learning_rate, args.weight_decay, args.latent_dims, args.mid_dims).to_string())
