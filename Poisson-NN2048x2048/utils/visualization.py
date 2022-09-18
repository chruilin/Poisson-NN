import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn
plt.style.use('classic') 
    
def losses_curve(Loss_all,save_path):
    ylabel_name = ['weight norm','loss_p', 'loss_div']
    plt.figure(figsize=[8,8])
    plt.subplots_adjust(hspace=0.3)
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(range(Loss_all.shape[0]), Loss_all[:,i])
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(ylabel_name[i])
        plt.grid()
    plt.savefig(save_path+'losses_curve.png')
    # plt.show() 

def calculate_curl(u, v, p, nx, ny, dx, dy):
    """
    u : nx+1 * ny
    v : nx * ny+1
    p : nx * ny
    u_y : nx * ny
    v_x : nx * ny
    
    uco ------ uco ------
     |          |          
     u   uce    u    uce   
     |          |          
    uco ------ uco ------
    
    vco ---v--- vco ---v---
     |           |          
     |     vce   |    vce   
     |           |          
    vco ---v--- vco ---v---
    """  
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx_ce, yy_ce = np.meshgrid(xce, yce, indexing='ij') 
    
    uce = (u[1:,:] + u[:-1,:])/2 # nx * ny
    u_y = np.zeros([nx,ny])
    u_y[:, 1:-1] = (uce[:,2:] - uce[:,:-2])/(2*dy)
    u_y[:, 0] = (uce[:,1] - uce[:,-1])/(2*dy)
    u_y[:, -1] = (uce[:,0] - uce[:,-2])/(2*dy)
   
    vce = (v[:,1:] + v[:,:-1])/2 # nx * ny
    v_x = np.zeros([nx,ny])
    v_x[1:-1,:] = (vce[2:,:] - vce[:-2,:])/(2*dx)
    v_x[0,:] = (vce[1,:] - vce[-1,:])/(2*dx)
    v_x[-1,:] = (vce[0,:] - vce[-2,:])/(2*dx)
    
    curl = v_x - u_y
    
    Values = np.stack((uce, vce, p, curl), axis=0)
    return Values  

def contourf_comparison(p, p_hat, nx, ny, dx, dy, save_path): 
    
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx, yy = np.meshgrid(xce, yce, indexing='ij')
      
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.contourf(xx, yy, p.reshape(xx.shape), levels=50, cmap=seaborn.cm.icefire)
    plt.colorbar(shrink=0.9)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$p_{DNS}$')
    
    plt.subplot(1,2,2)
    plt.contourf(xx, yy, p_hat.reshape(xx.shape), levels=50, cmap=seaborn.cm.icefire)
    plt.colorbar(shrink=0.9)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$\\hat{p}$')
      
    plt.savefig(save_path+'contourf_comparison.png')    
    # plt.show()
 


