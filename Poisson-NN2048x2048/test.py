# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:06:16 2022

@author: admin
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import scipy.io as sio
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs
from utils.models import PoissonNN
from utils.datasets import test_data, getLaplacianMatix
from utils.visualization import contourf_comparison
import argparse

def IterativeSolver(A4p, divUstar, p, method, tol, M=None):
    """
    Ax = rhs
    """
    num_iters = 0
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    if method.lower() == 'bicg':
        p, info = bicg(A4p, divUstar.reshape(-1), x0=p.reshape(-1), tol=tol, M=M, callback=callback)     
    if method.lower() == 'bicgstab':
        p, info = bicgstab(A4p, divUstar.reshape(-1), x0=p.reshape(-1), tol=tol, M=M, callback=callback) 
    if method.lower() == 'cg':
        p, info = cg(A4p, divUstar.reshape(-1), x0=p.reshape(-1), tol=tol, M=M, callback=callback) 
    if method.lower() == 'cgs':
        p, info = cgs(A4p, divUstar.reshape(-1), x0=p.reshape(-1), tol=tol, M=M, callback=callback)
    print('The iterative number is: %d' % num_iters)
    return p, num_iters

def MLSolver(model, A4p, divUstar, method, tol, M=None):  
    p = model(coordinates, divUstar)
    print('*'*10)
    p, iters = IterativeSolver(A4p, divUstar.cpu().numpy(), p.cpu().numpy()*0.001, method, tol, M)
    return p, iters
              
def run_test(model, useMLblock, method, tol):
    # model.load_state_dict(torch.load(save_dir+'/checkpoint/module.pt')) 
    
    # if not os.path.exists(save_dir+f'/Figures/{test_case}'):
    #     os.makedirs(save_dir+f'/Figures/{test_case}')
    
    total_iters = []
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        start_time = time.time() 
        for it in range(divUstar_test.shape[0]):
            if useMLblock:
                divUstar = torch.from_numpy(divUstar_test[it]).to(device)
                p_hat, num_iters = MLSolver(model, A4p, divUstar, method, tol)
            else:
                p_hat, num_iters = IterativeSolver(A4p, divUstar_test[it], p_n[it], method, tol)
            
            elapsed = time.time() - start_time
            start_time = time.time()  
            print('it: %d, t: %.3f, Time: %.3f' % (it, t_test[it], elapsed))
            total_iters.append(num_iters)
            # contourf_comparison(p_test[it]/0.0005, p_hat/0.0005, 
            #                     nx, ny, dx, dy, save_path=save_dir+f'/Figures/{test_case}/{it}')
    
    if not os.path.exists(save_dir+f'/Results/{test_case}/{method}_{tol}'):
        os.makedirs(save_dir+f'/Results/{test_case}/{method}_{tol}')
    
    sio.savemat(save_dir+f'/Results/{test_case}/{method}_{tol}/itersML{useMLblock}.mat',{'num_iters':np.array(total_iters),
                                                                                    't_test': t_test})
            
    return total_iters
                
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_case', type=str, default='k16TGV')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(device) 
    dtype = 'float64'
    if dtype == 'float32':
        ctype = torch.cfloat
    if dtype == 'float64':
        ctype = torch.cdouble

    # test_case = 'Decay32GRF'  # 'k16TGV', 'k32TGV', 'k16GRF', 'k32GRF', 'Decay16TGV', 'Decay32GRF'
    test_case = args.test_case
    nx = 2048  
    ny = 2048 
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny 
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx_ce, yy_ce = np.meshgrid(xce, yce, indexing='ij')   
    coordinates = np.stack((xx_ce, yy_ce))[np.newaxis].astype(dtype)
    coordinates = torch.from_numpy(coordinates).to(device)
    A4p = getLaplacianMatix(nx, ny, dx, dy)

    t_test = np.arange(0.5,10.5,0.5)
    
    divUstar_test, p_test, p_n = test_data(t_test, test_case) 
    divUstar_test = divUstar_test.astype(dtype)
    
    modes = 12
    hidden_units = 32
    activation = nn.GELU()
    model = PoissonNN(modes, modes, hidden_units, activation, ctype)   
    model.to(device)
    if dtype == 'float64':
        model.double()   
    print(model)
    
    save_dir = './Savers/PoissonNNfloat64'
    model.load_state_dict(torch.load(save_dir+'/checkpoint/module.pt')) 
    # method: bicg, bicgstab, cg, cgs, amg
    for method in ['bicgstab', 'bicg', 'cgs']:
        print(method)
        total_iters = run_test(model, useMLblock=True, method=method, tol=1e-6)
        total_iters = run_test(model, useMLblock=False, method=method, tol=1e-6)
        
   