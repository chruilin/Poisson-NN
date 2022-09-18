import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn
import scipy.io as sio
import os
plt.style.use('classic') 

def calculateVorticity(u, v, nx, ny, dx, dy):
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
    curl = np.sign(curl) * (np.abs(curl)/np.quantile(curl, 0.75))**0.5
    
    return curl  

def calculateDivU(p, dx, dy):
    p = np.concatenate((p[:,-1:], p, p[:,0:1]), axis=1)
    p = np.concatenate((p[:,:,-1:], p, p[:,:,0:1]), axis=2) 
       
    DivU = (p[:,0:-2,1:-1] - 2 * p[:,1:-1,1:-1] + p[:,2:,1:-1]) / dx**2 + \
        (p[:,1:-1,0:-2] - 2 * p[:,1:-1,1:-1] + p[:,1:-1,2:]) / dy**2     
    return DivU



def figure1():
    fig_dir = './PaperFigures/Figure1/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    nx = 2048  
    ny = 2048
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny     
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx, yy = np.meshgrid(xce, yce, indexing='ij')
    
    p = sio.loadmat('../Data/RandomDatan2048/p1.mat')['p'][::10] * 0.0005
    p = p - p.mean(axis=(1,2),keepdims=True)   
    divU = calculateDivU(p, dx, dy) 
    
    for i in range(5):
        with plt.style.context(['science', 'nature', 'no-latex']):
            plt.figure(figsize=(3,3))
            plt.contourf(xx, yy, p[i], levels=50, cmap=seaborn.cm.icefire)
            plt.axis('off')
            plt.savefig(fig_dir+f'p{i}.tif')
            plt.show()
   
    for i in range(5):
        with plt.style.context(['science', 'nature', 'no-latex']):
            plt.figure(figsize=(3,3))
            plt.contourf(xx, yy, divU[i], levels=50, cmap=seaborn.cm.icefire)
            plt.axis('off')  
            plt.savefig(fig_dir+f'b{i}.tif')
            plt.show()         
    
    
    path = '../../DataSet/KolmogorovFlow/CNAB/Re10000k16n2048CNAB_TGVinit/KolmogorovFlowRe10000k16n2048_t10.00000.mat'
    u = sio.loadmat(path)['u']
    v = sio.loadmat(path)['v'] 
    p = sio.loadmat(path)['p']
    Vorticity = calculateVorticity(u, v, nx, ny, dx, dy)
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(3,3))
        plt.contourf(xx, yy, Vorticity, levels=50, cmap=seaborn.cm.icefire)
        plt.axis('off')  
        plt.savefig(fig_dir+'Vorticity1.tif')
        plt.show()
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(3,3))
        plt.contourf(xx, yy, p, levels=50, cmap=seaborn.cm.icefire)
        plt.axis('off')  
        plt.savefig(fig_dir+'pdns1.tif')
        plt.show()

    path = '../../DataSet/DecayFlow/Re10000n2048CNAB_16TGVinit/DecayFlowRe10000n2048_t10.00000.mat'
    u = sio.loadmat(path)['u']
    v = sio.loadmat(path)['v'] 
    p = sio.loadmat(path)['p']
    Vorticity = calculateVorticity(u, v, nx, ny, dx, dy)
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(3,3))
        plt.contourf(xx, yy, Vorticity, levels=50, cmap=seaborn.cm.icefire)
        plt.axis('off')  
        plt.savefig(fig_dir+'Vorticity2.tif')
        plt.show()
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(3,3))
        plt.contourf(xx, yy, p, levels=50, cmap=seaborn.cm.icefire)
        plt.axis('off')  
        plt.savefig(fig_dir+'pdns2.tif')
        plt.show() 

def figure2():
    fig_dir = './PaperFigures/Figure2/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    path_f = ['./Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/bicgstab_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/bicg_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/cgs_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/bicgstab_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/bicg_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/cgs_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/bicgstab_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/bicg_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/cgs_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/bicgstab_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/bicg_1e-06/itersMLFalse.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/cgs_1e-06/itersMLFalse.mat'] 
    
    path_t = ['./Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/bicgstab_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/bicg_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16TGV/cgs_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/bicgstab_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/bicg_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32TGV/cgs_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/bicgstab_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/bicg_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k16GRF/cgs_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/bicgstab_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/bicg_1e-06/itersMLTrue.mat',
            './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results/k32GRF/cgs_1e-06/itersMLTrue.mat'] 

    nf, nt = [], []
    
    for path in path_f:
        n = sio.loadmat(path)['num_iters'][0]
        nf.append(n)
    for path in path_t:
        n = sio.loadmat(path)['num_iters'][0]
        nt.append(n)
        
    for i in range(12):
        print(i+1)
        alpha = nf[i] / nt[i] - 1
        print(alpha[1::2])
        print('*'*10)
    
    mean = 0
    for i in range(12):
        alpha = nf[i] / nt[i] - 1
        mean += np.mean(alpha)
    print(mean/12)
    # 3.97
        
    t_test = np.arange(0.5, 10.5, 0.5) 
    
  
    label_f = ['BICGSTAB',
              'BICG',
              'CGS']    
    label_t = ['Poisson-NN + BICGSTAB',
              'Poisson-NN + BICG',
              'Poisson-NN + CGS']
    
    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[10,10])    
        gs1 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.78, right=0.95, top=0.98, hspace=0.25, wspace=0.2)
        gs2 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.53, right=0.95, top=0.73, hspace=0.25, wspace=0.2) 
        gs3 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.28, right=0.95, top=0.48, hspace=0.25, wspace=0.2)
        gs4 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.03, right=0.95, top=0.23, hspace=0.25, wspace=0.2) 

        fig.text(0.03,0.87,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.37,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.12,'(d)', fontsize=12, fontweight='bold')
        
        for i in range(3):   
            ax1 = plt.subplot(gs1[0, i])
            ax1.plot(t_test, nf[i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs2[0, i])
            ax1.plot(t_test, nf[3+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[3+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs3[0, i])
            ax1.plot(t_test, nf[6+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[6+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs4[0, i])
            ax1.plot(t_test, nf[9+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[9+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)

        plt.savefig(fig_dir+'figure2.tif')
        plt.savefig(fig_dir+'figure2.svg')
        plt.show()         

def figure3():
    fig_dir = './PaperFigures/Figure3/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    nx = 1024  
    ny = 1024
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny     
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx, yy = np.meshgrid(xce, yce, indexing='ij')

    path_t2 = ['../../DataSet/KolmogorovFlow/CNAB/Re5000k16n1024CNAB_TGVinit/KolmogorovFlowRe5000k16n1024_t2.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB/Re5000k32n1024CNAB_TGVinit/KolmogorovFlowRe5000k32n1024_t2.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re5000k16n1024CNAB_GRFinit/KolmogorovFlowRe5000k16n1024_t2.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re5000k32n1024CNAB_GRFinit/KolmogorovFlowRe5000k32n1024_t2.0000.mat']
    
    path_t8 = ['../../DataSet/KolmogorovFlow/CNAB/Re5000k16n1024CNAB_TGVinit/KolmogorovFlowRe5000k16n1024_t8.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB/Re5000k32n1024CNAB_TGVinit/KolmogorovFlowRe5000k32n1024_t8.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re5000k16n1024CNAB_GRFinit/KolmogorovFlowRe5000k16n1024_t8.0000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re5000k32n1024CNAB_GRFinit/KolmogorovFlowRe5000k32n1024_t8.0000.mat']
    
    p_t2, omega_t2 = [], []
    for path in path_t2:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t2.append(p/0.0005)
        omega_t2.append(omega)
        
    p_t8, omega_t8 = [], []
    for path in path_t8:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t8.append(p/0.0005)
        omega_t8.append(omega)

    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[12,10])    
        gs1 = gridspec.GridSpec(4, 2, figure=fig, left=0.1, bottom=0.1, right=0.5, top=0.94, hspace=0.25, wspace=0.13)
        gs2 = gridspec.GridSpec(4, 2, figure=fig, left=0.55, bottom=0.1, right=0.95, top=0.94, hspace=0.25, wspace=0.14) 
  
        fig.text(0.15,0.95,'$p$'+' '+ '$(t=2)$', fontsize=12)
        fig.text(0.36,0.95,'$\\tilde{\omega}$'+' '+ '$(t=2)$', fontsize=12)      
        fig.text(0.60,0.95,'$p$'+' '+ '$(t=8)$', fontsize=12)
        fig.text(0.82,0.95,'$\\tilde{\omega}$'+' '+ '$(t=8)$', fontsize=12)
        
        fig.text(0.06,0.84,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.4,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.18,'(d)', fontsize=12, fontweight='bold')

        for i in range(4):   
            ax1 = plt.subplot(gs1[i, 0])
            cf1 = ax1.contourf(xx, yy, p_t2[i], levels=50, cmap=seaborn.cm.icefire) 
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs1[i, 1])
            cf1 = ax1.contourf(xx, yy, omega_t2[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
        for i in range(4):   
            ax1 = plt.subplot(gs2[i, 0])
            cf1 = ax1.contourf(xx, yy, p_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs2[i, 1])
            cf1 = ax1.contourf(xx, yy, omega_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)

        plt.savefig(fig_dir+'figure3.tif')
        plt.show()   

def figure4():
    fig_dir = './PaperFigures/Figure4/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    path_dir = './Poisson-NN2048x2048/Savers/PoissonNNfloat64/Results'
        
    path_f = [path_dir + '/k16TGV/bicgstab_1e-06/itersMLFalse.mat',
            path_dir + '/k16TGV/bicg_1e-06/itersMLFalse.mat',
            path_dir + '/k16TGV/cgs_1e-06/itersMLFalse.mat',
            path_dir + '/k32TGV/bicgstab_1e-06/itersMLFalse.mat',
            path_dir + '/k32TGV/bicg_1e-06/itersMLFalse.mat',
            path_dir + '/k32TGV/cgs_1e-06/itersMLFalse.mat',
            path_dir + '/k16GRF/bicgstab_1e-06/itersMLFalse.mat',
            path_dir + '/k16GRF/bicg_1e-06/itersMLFalse.mat',
            path_dir + '/k16GRF/cgs_1e-06/itersMLFalse.mat',
            path_dir + '/k32GRF/bicgstab_1e-06/itersMLFalse.mat',
            path_dir + '/k32GRF/bicg_1e-06/itersMLFalse.mat',
            path_dir + '/k32GRF/cgs_1e-06/itersMLFalse.mat'] 
    
    path_t = [path_dir + '/k16TGV/bicgstab_1e-06/itersMLTrue.mat',
            path_dir + '/k16TGV/bicg_1e-06/itersMLTrue.mat',
            path_dir + '/k16TGV/cgs_1e-06/itersMLTrue.mat',
            path_dir + '/k32TGV/bicgstab_1e-06/itersMLTrue.mat',
            path_dir + '/k32TGV/bicg_1e-06/itersMLTrue.mat',
            path_dir + '/k32TGV/cgs_1e-06/itersMLTrue.mat',
            path_dir + '/k16GRF/bicgstab_1e-06/itersMLTrue.mat',
            path_dir + '/k16GRF/bicg_1e-06/itersMLTrue.mat',
            path_dir + '/k16GRF/cgs_1e-06/itersMLTrue.mat',
            path_dir + '/k32GRF/bicgstab_1e-06/itersMLTrue.mat',
            path_dir + '/k32GRF/bicg_1e-06/itersMLTrue.mat',
            path_dir + '/k32GRF/cgs_1e-06/itersMLTrue.mat'] 

    nf, nt = [], []
    
    for path in path_f:
        n = sio.loadmat(path)['num_iters'][0]
        nf.append(n)
    for path in path_t:
        n = sio.loadmat(path)['num_iters'][0]
        nt.append(n)
        
    for i in range(12):
        print(i+1)
        alpha = nf[i] / nt[i] - 1
        print(alpha[1::2])
        print('*'*10)
    
    mean = 0
    for i in range(12):
        alpha = nf[i] / nt[i] - 1
        mean += np.mean(alpha)
    print(mean/12)
    # 2.23
        
    t_test = np.arange(0.5, 10.5, 0.5) 
    
  
    label_f = ['BICGSTAB',
              'BICG',
              'CGS']    
    label_t = ['Poisson-NN + BICGSTAB',
              'Poisson-NN + BICG',
              'Poisson-NN + CGS']
    
    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[10,10])    
        gs1 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.78, right=0.95, top=0.98, hspace=0.25, wspace=0.2)
        gs2 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.53, right=0.95, top=0.73, hspace=0.25, wspace=0.2) 
        gs3 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.28, right=0.95, top=0.48, hspace=0.25, wspace=0.2)
        gs4 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.03, right=0.95, top=0.23, hspace=0.25, wspace=0.2) 

        fig.text(0.03,0.87,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.37,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.12,'(d)', fontsize=12, fontweight='bold')
        
        for i in range(3):   
            ax1 = plt.subplot(gs1[0, i])
            ax1.plot(t_test, nf[i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs2[0, i])
            ax1.plot(t_test, nf[3+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[3+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs3[0, i])
            ax1.plot(t_test, nf[6+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[6+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs4[0, i])
            ax1.plot(t_test, nf[9+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[9+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)

        plt.savefig(fig_dir+'figure4.tif')
        plt.savefig(fig_dir+'figure4.svg')
        plt.show()         

    
def figure5():
    fig_dir = './PaperFigures/Figure5/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    nx = 2048  
    ny = 2048
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny     
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx, yy = np.meshgrid(xce, yce, indexing='ij')

    path_t2 = ['../../DataSet/KolmogorovFlow/CNAB/Re10000k16n2048CNAB_TGVinit/KolmogorovFlowRe10000k16n2048_t2.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB/Re10000k32n2048CNAB_TGVinit/KolmogorovFlowRe10000k32n2048_t2.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k16n2048CNAB_GRFinit/KolmogorovFlowRe10000k16n2048_t2.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k32n2048CNAB_GRFinit/KolmogorovFlowRe10000k32n2048_t2.00000.mat']
    
    path_t8 = ['../../DataSet/KolmogorovFlow/CNAB/Re10000k16n2048CNAB_TGVinit/KolmogorovFlowRe10000k16n2048_t8.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB/Re10000k32n2048CNAB_TGVinit/KolmogorovFlowRe10000k32n2048_t8.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k16n2048CNAB_GRFinit/KolmogorovFlowRe10000k16n2048_t8.00000.mat',
               '../../DataSet/KolmogorovFlow/CNAB_GRFinit/Re10000k32n2048CNAB_GRFinit/KolmogorovFlowRe10000k32n2048_t8.00000.mat']
    
    p_t2, omega_t2 = [], []
    for path in path_t2:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t2.append(p/0.00025)
        omega_t2.append(omega)
        
    p_t8, omega_t8 = [], []
    for path in path_t8:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t8.append(p/0.00025)
        omega_t8.append(omega)

    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[12,10])    
        gs1 = gridspec.GridSpec(4, 2, figure=fig, left=0.1, bottom=0.1, right=0.5, top=0.94, hspace=0.25, wspace=0.14)
        gs2 = gridspec.GridSpec(4, 2, figure=fig, left=0.55, bottom=0.1, right=0.95, top=0.94, hspace=0.25, wspace=0.14) 
  
        fig.text(0.15,0.95,'$p$'+' '+ '$(t=2)$', fontsize=12)
        fig.text(0.36,0.95,'$\\tilde{\omega}$'+' '+ '$(t=2)$', fontsize=12)      
        fig.text(0.60,0.95,'$p$'+' '+ '$(t=8)$', fontsize=12)
        fig.text(0.82,0.95,'$\\tilde{\omega}$'+' '+ '$(t=8)$', fontsize=12)
        
        fig.text(0.06,0.84,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.4,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.18,'(d)', fontsize=12, fontweight='bold')
       

        for i in range(4):   
            ax1 = plt.subplot(gs1[i, 0])
            cf1 = ax1.contourf(xx, yy, p_t2[i], levels=50, cmap=seaborn.cm.icefire) 
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs1[i, 1])
            cf1 = ax1.contourf(xx, yy, omega_t2[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
        for i in range(4):   
            ax1 = plt.subplot(gs2[i, 0])
            cf1 = ax1.contourf(xx, yy, p_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs2[i, 1])
            cf1 = ax1.contourf(xx, yy, omega_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)

        plt.savefig(fig_dir+'figure5.tif')
        plt.show()  

def figure6():
    fig_dir = './PaperFigures/Figure6/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    path_dir1024 = './Poisson-NN1024x1024/Savers/PoissonNNfloat64/Results'    
    path_dir2048 = './Poisson-NN2048x2048/Savers/PoissonNNfloat64/Results'
        
    path_f = [path_dir1024 + '/Decay16TGV/bicgstab_1e-06/itersMLFalse.mat',
            path_dir1024 + '/Decay16TGV/bicg_1e-06/itersMLFalse.mat',
            path_dir1024 + '/Decay16TGV/cgs_1e-06/itersMLFalse.mat',
            path_dir1024 + '/Decay32GRF/bicgstab_1e-06/itersMLFalse.mat',
            path_dir1024 + '/Decay32GRF/bicg_1e-06/itersMLFalse.mat',
            path_dir1024 + '/Decay32GRF/cgs_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay16TGV/bicgstab_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay16TGV/bicg_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay16TGV/cgs_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay32GRF/bicgstab_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay32GRF/bicg_1e-06/itersMLFalse.mat',
            path_dir2048 + '/Decay32GRF/cgs_1e-06/itersMLFalse.mat'] 
    
    path_t = [path_dir1024 + '/Decay16TGV/bicgstab_1e-06/itersMLTrue.mat',
            path_dir1024 + '/Decay16TGV/bicg_1e-06/itersMLTrue.mat',
            path_dir1024 + '/Decay16TGV/cgs_1e-06/itersMLTrue.mat',
            path_dir1024 + '/Decay32GRF/bicgstab_1e-06/itersMLTrue.mat',
            path_dir1024 + '/Decay32GRF/bicg_1e-06/itersMLTrue.mat',
            path_dir1024 + '/Decay32GRF/cgs_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay16TGV/bicgstab_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay16TGV/bicg_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay16TGV/cgs_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay32GRF/bicgstab_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay32GRF/bicg_1e-06/itersMLTrue.mat',
            path_dir2048 + '/Decay32GRF/cgs_1e-06/itersMLTrue.mat'] 

    nf, nt = [], []
    
    for path in path_f:
        n = sio.loadmat(path)['num_iters'][0]
        nf.append(n)
    for path in path_t:
        n = sio.loadmat(path)['num_iters'][0]
        nt.append(n)
        
    for i in range(12):
        print(i+1)
        alpha = nf[i] / nt[i] - 1
        print(alpha[1::2])
        print('*'*10)
    
    mean = 0
    for i in range(6):
        alpha = nf[i] / nt[i] - 1
        mean += np.mean(alpha)
    print(mean/6)
    # 2.15
    
    mean = 0
    for i in range(6):
        alpha = nf[i+6] / nt[i+6] - 1
        mean += np.mean(alpha)
    print(mean/6)
        
    t_test = np.arange(0.5, 10.5, 0.5) 
    
  
    label_f = ['BICGSTAB',
              'BICG',
              'CGS']    
    label_t = ['Poisson-NN + BICGSTAB',
              'Poisson-NN + BICG',
              'Poisson-NN + CGS']
    
    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[10,10])    
        gs1 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.78, right=0.95, top=0.98, hspace=0.25, wspace=0.2)
        gs2 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.53, right=0.95, top=0.73, hspace=0.25, wspace=0.2) 
        gs3 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.28, right=0.95, top=0.48, hspace=0.25, wspace=0.2)
        gs4 = gridspec.GridSpec(1, 3, figure=fig, left=0.12, bottom=0.03, right=0.95, top=0.23, hspace=0.25, wspace=0.2) 

        fig.text(0.03,0.87,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.37,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.03,0.12,'(d)', fontsize=12, fontweight='bold')
        
        for i in range(3):   
            ax1 = plt.subplot(gs1[0, i])
            ax1.plot(t_test, nf[i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs2[0, i])
            ax1.plot(t_test, nf[3+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[3+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs3[0, i])
            ax1.plot(t_test, nf[6+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[6+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)
        
        for i in range(3):   
            ax1 = plt.subplot(gs4[0, i])
            ax1.plot(t_test, nf[9+i], 'k-o', label=label_f[i])
            ax1.plot(t_test, nt[9+i], 'r-o', label=label_t[i])
            ax1.set_xlabel('$t$', fontsize=10)
            if i == 0:
                ax1.set_ylabel('Iteration number $n$', fontsize=11)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(loc='center left', fontsize=10)

        plt.savefig(fig_dir+'figure6.tif')
        plt.savefig(fig_dir+'figure6.svg')
        plt.show()

def figure7():
    fig_dir = './PaperFigures/Figure7/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    nx = 1024  
    ny = 1024
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny     
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx1024, yy1024 = np.meshgrid(xce, yce, indexing='ij')

    path_t2 = ['../../DataSet/DecayFlow/Re5000n1024CNAB_16TGVinit/DecayFlowRe5000n1024_t2.0000.mat',
               '../../DataSet/DecayFlow/Re5000n1024CNAB_32GRFinit/DecayFlowRe5000n1024_t2.0000.mat']
    
    path_t8 = ['../../DataSet/DecayFlow/Re5000n1024CNAB_16TGVinit/DecayFlowRe5000n1024_t8.0000.mat',
               '../../DataSet/DecayFlow/Re5000n1024CNAB_32GRFinit/DecayFlowRe5000n1024_t8.0000.mat']
    
    p_t2, omega_t2 = [], []
    p_t8, omega_t8 = [], []
    
    for path in path_t2:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t2.append(p/0.0005)
        omega_t2.append(omega)

    for path in path_t8:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t8.append(p/0.0005)
        omega_t8.append(omega)
        
    nx = 2048  
    ny = 2048
    dx = 2*np.pi/nx 
    dy = 2*np.pi/ny     
    xce = (np.arange(1,nx+1) - 0.5) * dx
    yce = (np.arange(1,ny+1) - 0.5) * dy   
    xx2048, yy2048 = np.meshgrid(xce, yce, indexing='ij')

    path_t2 = ['../../DataSet/DecayFlow/Re10000n2048CNAB_16TGVinit/DecayFlowRe10000n2048_t2.00000.mat',
               '../../DataSet/DecayFlow/Re10000n2048CNAB_32GRFinit/DecayFlowRe10000n2048_t2.00000.mat']
    
    path_t8 = ['../../DataSet/DecayFlow/Re10000n2048CNAB_16TGVinit/DecayFlowRe10000n2048_t8.00000.mat',
               '../../DataSet/DecayFlow/Re10000n2048CNAB_32GRFinit/DecayFlowRe10000n2048_t8.00000.mat']
    
    for path in path_t2:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t2.append(p/0.00025)
        omega_t2.append(omega)
        
    for path in path_t8:
        u = sio.loadmat(path)['u']
        v = sio.loadmat(path)['v']
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        omega = calculateVorticity(u, v, nx, ny, dx, dy)
        p_t8.append(p/0.00025)
        omega_t8.append(omega)

    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = plt.figure(figsize=[12,10])    
        gs1 = gridspec.GridSpec(4, 2, figure=fig, left=0.1, bottom=0.1, right=0.5, top=0.94, hspace=0.25, wspace=0.14)
        gs2 = gridspec.GridSpec(4, 2, figure=fig, left=0.55, bottom=0.1, right=0.95, top=0.94, hspace=0.25, wspace=0.14) 
  
        fig.text(0.15,0.95,'$p$'+' '+ '$(t=2)$', fontsize=12)
        fig.text(0.36,0.95,'$\\tilde{\omega}$'+' '+ '$(t=2)$', fontsize=12)      
        fig.text(0.60,0.95,'$p$'+' '+ '$(t=8)$', fontsize=12)
        fig.text(0.82,0.95,'$\\tilde{\omega}$'+' '+ '$(t=8)$', fontsize=12)
        
        fig.text(0.06,0.84,'(a)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.62,'(b)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.4,'(c)', fontsize=12, fontweight='bold')
        fig.text(0.06,0.18,'(d)', fontsize=12, fontweight='bold')
       

        for i in range(4):   
            ax1 = plt.subplot(gs1[i, 0])
            if i < 2:
                cf1 = ax1.contourf(xx1024, yy1024, p_t2[i], levels=50, cmap=seaborn.cm.icefire) 
            else:
                cf1 = ax1.contourf(xx2048, yy2048, p_t2[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs1[i, 1])
            if i < 2:
                cf1 = ax1.contourf(xx1024, yy1024, omega_t2[i], levels=50, cmap=seaborn.cm.icefire)
            else:
                cf1 = ax1.contourf(xx2048, yy2048, omega_t2[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
        for i in range(4):   
            ax1 = plt.subplot(gs2[i, 0])
            if i < 2:
                cf1 = ax1.contourf(xx1024, yy1024, p_t8[i], levels=50, cmap=seaborn.cm.icefire) 
            else:
                cf1 = ax1.contourf(xx2048, yy2048, p_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            
            ax1 = plt.subplot(gs2[i, 1])
            if i < 2:
                cf1 = ax1.contourf(xx1024, yy1024, omega_t8[i], levels=50, cmap=seaborn.cm.icefire)
            else:
                cf1 = ax1.contourf(xx2048, yy2048, omega_t8[i], levels=50, cmap=seaborn.cm.icefire)
            cbar = plt.colorbar(cf1,shrink=0.9)
            cbar.ax.tick_params(labelsize=9) 
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)

        plt.savefig(fig_dir+'figure7.tif')
        plt.show()  


figure2()
figure3() 
figure4()
figure5()
figure6()   
figure7()     
 