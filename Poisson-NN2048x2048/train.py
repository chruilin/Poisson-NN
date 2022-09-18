# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:55:36 2022

@author: admin
"""


import numpy as np
import torch
import torch.nn as nn
import time
import os
import scipy.io as sio
from utils.models import PoissonNN
from utils.datasets import read_training_data
from utils.visualization import losses_curve, contourf_comparison
import argparse

def get_weights_norm(model):
    weights_norm = 0.
    for name, param in model.named_parameters():
        # print(name,param.shape,param)
        if 'weight' in name:
            weights_norm += param.norm(2)**2
    return weights_norm

def Div_loss(p, divUstar):    
    p = torch.cat((p[:,:,-1:], p, p[:,:,0:1]), dim=2)
    p = torch.cat((p[:,:,:,-1:], p, p[:,:,:,0:1]), dim=3)       
    Lp = (p[:,:,0:-2,1:-1] - 2 * p[:,:,1:-1,1:-1] + p[:,:,2:,1:-1]) / dx**2 + \
        (p[:,:,1:-1,0:-2] - 2 * p[:,:,1:-1,1:-1] + p[:,:,1:-1,2:]) / dy**2   
    loss_div = torch.mean(torch.norm(Lp[:,0] - divUstar[:,0], dim=(1,2))**2)   
    return loss_div

def train_step(model, divUstar, p): 
    p_hat = model(coordinates, divUstar)
    loss_div = Div_loss(p_hat*0.001, divUstar)
    loss_p = torch.mean(torch.norm(p_hat[:,0] - p, dim=(1,2))**2)
    loss = loss_div + loss_p
    return loss, loss_p, loss_div

def run_train(model, optimizer, num_epochs, restore=False):
    if not os.path.exists(save_dir+'/checkpoint/'):
        os.makedirs(save_dir+'/checkpoint/')
    if not os.path.exists(save_dir+'/Figures/'):
        os.makedirs(save_dir+'/Figures/')
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[500,1000,1500,2000], gamma=0.5)
    
    if restore:
        loss_all = sio.loadmat(save_dir+'/loss_all.mat')['loss_all'].tolist()
        model.load_state_dict(torch.load(save_dir+'/checkpoint/module.pt')) 
        model.train()
        
    else:
        loss_all = []
   
    print('Training......')  
    model.train()
    start_time = time.time() 
    weights_norm = get_weights_norm(model)
    for epoch in range(1,num_epochs+1): 
        loss_p_it, loss_div_it = [], []     
        index = np.random.permutation(divUstar_train[0].shape[0])
        for k in range(5):
            for it in range(len(index)//25):
                index_it = index[it*25:(it+1)*25]
                divUstar = torch.from_numpy(divUstar_train[k][index_it][:,np.newaxis]).to(device)
                p = torch.from_numpy(p_train[k][index_it]).to(device)
                    
                optimizer.zero_grad()  
      
                loss, loss_p, loss_div = train_step(model, divUstar, p)
    
                loss.backward()
                optimizer.step()
    
                loss_p_it.append(loss_p.item())
                loss_div_it.append(loss_div.item())
            
                if it % 1 == 0:
                    elapsed = time.time() - start_time
                    print('Epoch: %d, k: %d, step: %d, Time: %.3f, weights_norm: %.3e, loss_p: %.3e, loss_div: %.3e'
                          % (epoch, k, it, elapsed, weights_norm, np.mean(loss_p_it), np.mean(loss_div_it)))                
                    start_time = time.time()  
            
        
        scheduler.step()
        weights_norm = get_weights_norm(model)  
        print('***********************************************************')           
        loss_all.append([weights_norm.detach().cpu().numpy(), np.mean(loss_p_it), np.mean(loss_div_it)])
        
        if epoch % 1 == 0:
            torch.save(model.state_dict(), save_dir+'/checkpoint/module.pt')            
            sio.savemat(save_dir+'/loss_all.mat',{'loss_all':np.array(loss_all)})
        
        if epoch % 100 ==0: 
            losses_curve(np.array(loss_all),save_dir+'/Figures/')
            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                for it in range(divUstar_test.shape[0]):
                    divUstar = torch.from_numpy(divUstar_test[it:it+1][:,np.newaxis]).to(device)
                    
                    p_hat = model(coordinates, divUstar)
                    contourf_comparison(p_test[it], p_hat.cpu().numpy(), 
                                        nx, ny, dx, dy, save_path=save_dir+f'/Figures/{it}')

            model.train()   


               
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='float32')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(device)  
    dtype = args.dtype
    if dtype == 'float32':
        ctype = torch.cfloat
    if dtype == 'float64':
        ctype = torch.cdouble

    save_dir = './Savers/PoissonNNfloat64'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nx = 2048  
    ny = 2048 
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny 
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx_ce, yy_ce = np.meshgrid(xce, yce, indexing='ij')
    coordinates = np.stack((xx_ce, yy_ce))[np.newaxis].astype(dtype)
    coordinates = torch.from_numpy(coordinates).to(device)
    del xx_ce, yy_ce, xce, yce

    divUstar_train, p_train = read_training_data(dtype) 
    divUstar_test, p_test = divUstar_train[::100], p_train[::100] 

    modes = 12
    hidden_units = 32
    activation = nn.GELU()
    model = PoissonNN(modes, modes, hidden_units, activation, ctype) 
    
    model.to(device)
    if dtype == 'float64':
        model.double()
    print(model)

    if dtype == 'float32':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        run_train(model, optimizer, num_epochs=2000, restore=False)
    if dtype == 'float64':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000625, weight_decay=1e-5)
        run_train(model, optimizer, num_epochs=500, restore=True)
    