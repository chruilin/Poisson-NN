# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:26:03 2022

@author: admin
"""


import numpy as np
import scipy.io as sio
from scipy import sparse
import torch.nn as nn
import torch

def getLaplacianMatix(nx, ny, dx, dy):    
    d = -2. * np.ones([nx,])
    up = np.ones([nx-1,]) 
    L4x = np.diag(d) + np.diag(up, 1) + np.diag(up, -1)
    L4x[0,-1] = 1. 
    L4x[-1,0] = 1. 
    L4x = L4x / dx**2
    
    d = -2. * np.ones([ny,])
    up = np.ones([ny-1,]) 
    L4y = np.diag(d) + np.diag(up, 1) + np.diag(up, -1)
    L4y[0,-1] = 1.
    L4y[-1,0] = 1.
    L4y = L4y / dy**2 
    
    A4p = sparse.kron(sparse.eye(ny), L4x) + sparse.kron(L4y, sparse.eye(nx))    
    A4p = A4p.tocoo()
    return A4p

def divUstar_back(u, v, p, dx, dy):
    """
    u : (nx+1, ny)
    v : (nx, ny+1)
    p : (nx, ny)
    
    u* = u + Gp
    
    u = u* - Gp'
    """ 
    
    u[1:-1,:] = u[1:-1,:] + (p[1:,:] - p[0:-1,:])/dx 
    u[0,:] = u[0,:] + (p[0,:] - p[-1,:])/dx
    u[-1,:] = u[0,:]
    
    v[:,1:-1] = v[:,1:-1] + (p[:,1:] - p[:,0:-1])/dy
    v[:,0] = v[:,0] + (p[:,0] - p[:,-1])/dy
    v[:,-1] = v[:,0]
    
    divUstar = (u[1:,:] - u[:-1,:])/dx + (v[:,1:] - v[:,:-1])/dy # (nx, ny)

    return divUstar

def calculateDivU(p, dx, dy):
    
    p = np.concatenate((p[:,-1:], p, p[:,0:1]), axis=1)
    p = np.concatenate((p[:,:,-1:], p, p[:,:,0:1]), axis=2) 
       
    divU = (p[:,0:-2,1:-1] - 2 * p[:,1:-1,1:-1] + p[:,2:,1:-1]) / dx**2 + \
        (p[:,1:-1,0:-2] - 2 * p[:,1:-1,1:-1] + p[:,1:-1,2:]) / dy**2
        
    return divU

def avg_filter(p):
    p = np.concatenate((p[:,-1:], p, p[:,0:1]), axis=1)
    p = np.concatenate((p[:,:,-1:], p, p[:,:,0:1]), axis=2)  # (nx+2, ny+2)      
    p = nn.AvgPool2d(3, stride=1)(torch.from_numpy(p)).numpy() 
    return p

def datafilter():
    for k in range(1,6):
        p = sio.loadmat(f'../../Data/RandomData2048x2048/p{k}.mat')['p'] 
        for i in range(50):
            p = avg_filter(p)
        sio.savemat(f'../../Data/RandomData2048x2048/p{k}.mat',{'p':p})

def read_training_data(dtype):    
    nx = 2048  
    ny = 2048
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny  
     
    divUstar, p_out = [], []
    for k in range(1,6):
        p = sio.loadmat(f'../../Data/RandomData2048x2048/p{k}.mat')['p'] * 0.00025 
        p = p - p.mean(axis=(1,2),keepdims=True)
        divU = calculateDivU(p, dx, dy)      
        divUstar.append(divU.astype(dtype))
        p = p / 0.001
        p_out.append(p.astype(dtype))
        
    return divUstar, p_out

def test_data(t_test, case):
    nx = 2048  
    ny = 2048
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny 
    
    divUstar, p_out = [], []
    p_n = []
    
    if case == 'k16TGV':
        dir_path = '../../../DataSet/KolmogorovFlow/CNAB/Re10000k16n2048CNAB_TGVinit/KolmogorovFlowRe10000k16n2048'
    if case == 'k32TGV':
        dir_path = '../../../DataSet/KolmogorovFlow/CNAB/Re10000k32n2048CNAB_TGVinit/KolmogorovFlowRe10000k32n2048'
    if case == 'k16GRF':
        dir_path = '../../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k16n2048CNAB_GRFinit/KolmogorovFlowRe10000k16n2048'
    if case == 'k32GRF':
        dir_path = '../../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k32n2048CNAB_GRFinit/KolmogorovFlowRe10000k32n2048'
    
    if case == 'Decay16TGV':
        dir_path = '../../../DataSet/DecayFlow/Re10000n2048CNAB_16TGVinit/DecayFlowRe10000n2048'    
    if case == 'Decay32TGV':
        dir_path = '../../../DataSet/DecayFlow/Re10000n2048CNAB_32TGVinit/DecayFlowRe10000n2048'
    
    if case == 'Decay16GRF':
        dir_path = '../../../DataSet/DecayFlow/Re10000n2048CNAB_16GRFinit/DecayFlowRe10000n2048'    
    if case == 'Decay32GRF':
        dir_path = '../../../DataSet/DecayFlow/Re10000n2048CNAB_32GRFinit/DecayFlowRe10000n2048'

    for t in t_test-0.00025:  
        path = dir_path + f'_t{t:.5f}.mat'
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        p_n.append(p)
        
    for t in t_test:  
        path = dir_path + f'_t{t:.5f}.mat'
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v'] 
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        divU = divUstar_back(u, v, p, dx, dy)
        divUstar.append(divU)
        p_out.append(p)

    p_n = np.stack(p_n,axis=0)   
    divUstar = np.stack(divUstar,axis=0)
    p_out = np.stack(p_out,axis=0)
    
    return divUstar, p_out, p_n