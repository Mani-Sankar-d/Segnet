import torch 
import torch.nn as nn

class Segnet(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,x):
        x,indices,sizes = self.encoder(x)
        return self.decoder(x,indices,sizes)