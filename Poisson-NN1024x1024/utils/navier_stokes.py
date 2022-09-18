# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:47:07 2022

@author: admin
"""

import numpy as np
from scipy import sparse
import torch

def periodicBC(u, v):
    """
    u : (nx+1, ny) --> (nx+3, ny+2)
    v : (nx, ny+1) --> (nx+2, ny+3)
    """
    u = torch.cat((u[-2:-1], u, u[1:2]), dim=0)
    u = torch.cat((u[:,-1:], u, u[:,0:1]), dim=1)
    
    v = torch.cat((v[-1:], v, v[0:1]), dim=0)
    v = torch.cat((v[:,-2:-1], v, v[:,1:2]), dim=1)
 
    return u, v
    
def viscous_term(ubig, vbig, dx, dy):
    """
    ubig : (nx+3, ny+2)
    vbig : (nx+2, ny+3) 
    
    Explicit Viscous terms, Lu(n) + bc(n)
    
    Lu(i,j) = (u(i-1,j) - 2u(i,j) + u(i+1,j))/dx**2 + (u(i,j-1) - 2u(i,j) + u(i,j+1))/dy**2
    Lv(i,j) = (v(i-1,j) - 2v(i,j) + v(i+1,j))/dx**2 + (v(i,j-1) - 2v(i,j) + v(i,j+1))/dy**2
    """   
    Lu = (ubig[0:-2,1:-1] - 2*ubig[1:-1,1:-1] + ubig[2:,1:-1])/dx**2 + \
          (ubig[1:-1,0:-2] - 2*ubig[1:-1,1:-1] + ubig[1:-1,2:])/dy**2    # nx+1 * ny
    
    Lv = (vbig[0:-2,1:-1] - 2*vbig[1:-1,1:-1] + vbig[2:,1:-1])/dx**2 + \
         (vbig[1:-1,0:-2] - 2*vbig[1:-1,1:-1] + vbig[1:-1,2:])/dy**2     # nx * ny+1            
    return Lu, Lv

def convective_term(ubig, vbig, dx, dy):
    """
    ubig : (nx+3, ny+2)
    vbig : (nx+2, ny+3) 
    
    Nu(i,j) = (uce(i,j)**2 - uce(i-1,j)**2)/dx + (uco(i,j+1) * vco(i,j+1) - uco(i,j) * vco(i,j))/dy
    Nv(i,j) = (uco(i+1,j) * vco(i+1,j) - uco(i,j) * vco(i,j))/dx + (vce(i,j)**2 - vce(i,j-1)**2)/dy
    """ 
    # 1. interpolate velocity at cell center/cell cornder  
    uce = (ubig[0:-1, 1:-1] + ubig[1:, 1:-1])/2     # nx+2 * ny
    uco = (ubig[1:-1, 0:-1] + ubig[1:-1, 1:])/2     # nx+1 * ny+1
    vce = (vbig[1:-1, 0:-1] + vbig[1:-1, 1:])/2     # nx * ny+2
    vco = (vbig[0:-1, 1:-1] + vbig[1:, 1:-1])/2     # nx+1 * ny+1
    
    # 2. multiply
    uuce = uce * uce    # nx+2 * ny
    uvco = uco * vco    # nx+1 * ny+1
    vvce = vce * vce    # nx * ny+2
    
    # 3. get derivative for u, v
    Nu = (uuce[1:, :] - uuce[0:-1, :])/dx + (uvco[:, 1:] - uvco[:, 0:-1])/dy  # nx+1 * ny
    Nv = (uvco[1:, :] - uvco[0:-1, :])/dx + (vvce[:, 1:] - vvce[:, 0:-1])/dy  # nx * ny+1        
    return Nu, Nv

def momentumEqRHS(u, v, uForce, vForce, dx, dy, Re):
    """
    u : (nx, ny)
    v : (nx, ny)
    uForce : (nx+1, ny)
    
    output:
    rhs_u : (nx+1, ny) 
    rhs_v : (nx, ny+1)
    """
    # Periodic boundary conditions       
    u, v = periodicBC(u, v)
    
    # Viscous terms      
    Lu, Lv = viscous_term(u, v, dx, dy)
            
    # Convective terms
    Nu, Nv = convective_term(u, v, dx, dy)
    if uForce is not None:
        Nu = Nu - uForce
    if vForce is not None:
        Nv = Nv - vForce

    rhs_u = -Nu + Lu/Re
    rhs_v = -Nv + Lv/Re
    
    return rhs_u, rhs_v
    
def velocity_correction(ustar, vstar, p, dx, dy):
    """
    u : (nx+1, ny)
    v : (nx, ny+1)
    p : (nx, ny)
    
    u = u* - Gp'
    """ 
    u0 = ustar[0:1,:] - (p[0:1,:] - p[-1:,:])/dx # 1 * ny
    u = ustar[1:-1,:] - (p[1:,:] - p[0:-1,:])/dx # nx-1 * ny   
    u = torch.cat((u0, u, u0), dim=0) # (nx+1, ny)
    
    v0 = vstar[:,0:1] - (p[:,0:1] - p[:,-1:])/dy # nx * 1
    v = vstar[:,1:-1] - (p[:,1:] - p[:,0:-1])/dy # nx * ny-1
    v = torch.cat((v0, v, v0), dim=1) # (nx, ny+1) 
    
    return u, v  

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
    invD = sparse.diags(1./A4p.diagonal()).tocoo()
    return A4p, invD