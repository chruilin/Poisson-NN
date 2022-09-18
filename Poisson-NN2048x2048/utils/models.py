# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:00:48 2022

@author: admin
"""

import torch.nn as nn
import torch

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, ctype):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.dtype = ctype

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=self.dtype))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=self.dtype))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=self.dtype, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, ctype):
        super(SpectralConv, self).__init__()       
        self.conv = SpectralConv2d(in_channels, out_channels, modes1, modes2, ctype)
        self.w = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2       
        return x

class GreenConv(nn.Module):
    def __init__(self,):
        super(GreenConv, self).__init__()  
        
    def forward(self, g, h):
        g_ft = torch.fft.rfft2(g)
        h_ft = torch.fft.rfft2(h)
        x = g_ft * h_ft
        x = torch.fft.irfft2(x, s=(h.size(-2), h.size(-1)))
        return x
  
class PoissonNN(nn.Module):
    def __init__(self, modes1, modes2,  hidden_units, activation, ctype):
        super(PoissonNN, self).__init__()
        
        self.g = nn.Sequential(nn.Conv2d(4, hidden_units, 1),
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            nn.Conv2d(hidden_units, 1, 1))
        self.uhom = nn.Sequential(nn.Conv2d(2, hidden_units, 1),
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            SpectralConv(hidden_units, hidden_units, modes1, modes2, ctype), activation,
            nn.Conv2d(hidden_units, 1, 1))
        
        self.greenconv = GreenConv()

    def forward(self, x, h):
        g = self.g(torch.cat((x,x),dim=1))
        u = self.uhom(x)
        y = self.greenconv(g, h) + u
        return y     
