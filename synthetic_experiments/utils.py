import numpy as np
import matplotlib.pyplot as plt

def normalize(v, r):
    return v/np.sqrt(np.sum(v**2))*r

def viz_pc(pc):
    
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    xdata = pc[:,0]
    ydata = pc[:,1]
    zdata = pc[:,2]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    
    return

def viz_all_pc(pc_g,pc_o,pc_p, name = 'shape'):
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    xdata = pc_g[:,0]
    ydata = pc_g[:,1]
    zdata = pc_g[:,2]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/" + name +"_gauss.png"
    plt.savefig(fig_name, transparent=True)
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    xdata = pc_o[:,0]
    ydata = pc_o[:,1]
    zdata = pc_o[:,2]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/" + name +"_ortho.png"
    plt.savefig(fig_name, transparent=True)
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    xdata = pc_p[:,0]
    ydata = pc_p[:,1]
    zdata = pc_p[:,2]
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/" + name +"_plane.png"
    plt.savefig(fig_name, transparent=True)
    
    return

def viz_for_paper(pc, pc_g, pc_p, pc_o, H = None, name = 'shape', plot_size = 10):
    
    fig = plt.figure(figsize = (plot_size*4,plot_size))
    
    ax0 = fig.add_subplot(141, projection='3d')
    xdata = pc[:,0]
    ydata = pc[:,1]
    zdata = pc[:,2]
    ax0.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax0.axis('off')
    
    ax1 = fig.add_subplot(142, projection='3d')
    xdata = pc_g[:,0]
    ydata = pc_g[:,1]
    zdata = pc_g[:,2]
    ax1.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax1.axis('off')
    

    ax2 = fig.add_subplot(143, projection='3d')
    xdata = pc_p[:,0]
    ydata = pc_p[:,1]
    zdata = pc_p[:,2]
    ax2.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax2.axis('off')

    ax3 = fig.add_subplot(144, projection='3d')
    xdata = pc_o[:,0]
    ydata = pc_o[:,1]
    zdata = pc_o[:,2]
    ax3.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax3.axis('off')
    
    if H != None:
        if name == "line":
            t1 = "H0: " + "{:.4f}".format(H[0]) 
            ax1.set_title(t1, y= 0.9, pad = 0)
            ax1.title.set_size(plot_size*2)
            t2 = "H0: " + "{:.4f}".format(H[2])
            ax2.set_title(t2, y= 0.9, pad = 0)
            ax2.title.set_size(plot_size*2)
            t3 = "H0: " + "{:.4f}".format(H[4])
            ax3.set_title(t3, y= 0.9, pad = 0)
            ax3.title.set_size(plot_size*2)
        else:
            t1 = "H0: " + "{:.4f}".format(H[0]) + " | H1: " + "{:.4f}".format(H[1])
            ax1.set_title(t1, y= 0.9, pad = 0)
            ax1.title.set_size(plot_size*2)
            t2 = "H0: " + "{:.4f}".format(H[2]) + " | H1: " + "{:.4f}".format(H[3])
            ax2.set_title(t2, y= 0.9, pad = 0)
            ax2.title.set_size(plot_size*2)
            t3 = "H0: " + "{:.4f}".format(H[4]) + " | H1: " + "{:.4f}".format(H[5])
            ax3.set_title(t3, y= 0.9, pad = 0)
            ax3.title.set_size(plot_size*2)
        
    fig_name = "viz/" + name +".png"
    plt.savefig(fig_name, transparent=True)
    
    return