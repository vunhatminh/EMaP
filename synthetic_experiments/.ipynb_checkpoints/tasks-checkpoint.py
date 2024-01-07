import numpy as np
import matplotlib.pyplot as plt
import manifold_synthetic
import utils
import csv
import pickle

from ripser import Rips
from persim import plot_diagrams, bottleneck

def synthetic(args):
    print("---Synthetic experiment")
    if args.syn_sub == "circle":
        xdata, ydata, zdata, fig_name = gen_circle(args)
    elif args.syn_sub == "spiral":
        xdata, ydata, zdata, fig_name = gen_spiral(args)
    elif args.syn_sub == "spiral2":
        xdata, ydata, zdata, fig_name = gen_spiral2(args)
    elif args.syn_sub == "spiral3":
        xdata, ydata, zdata, fig_name = gen_spiral3(args)
    elif args.syn_sub == "line":
        xdata, ydata, zdata, fig_name = gen_line(args)
    elif args.syn_sub == "circle2":
        xdata, ydata, zdata, fig_name = gen_circle2(args)
    elif args.syn_sub == "circle3":
        xdata, ydata, zdata, fig_name = gen_circle3(args)
    else:
        print("Not valid setting")
    
    print("--Visualization of the dataset is saved at: ", fig_name)

    data = np.transpose(np.concatenate(([xdata],[ydata],[zdata]), axis=0))
    if args.syn_sub == "line":
        manifold_sampler = manifold_synthetic.Manifold_Synthetic_Sampler(data, dim = 1)
    else:
        manifold_sampler = manifold_synthetic.Manifold_Synthetic_Sampler(data)
    manifold_G = manifold_sampler.get_G_from_data()
    Gu, Gd, Gv = np.linalg.svd(manifold_G, full_matrices=False)
    
    bottleneck_results = []
    
    for _ in range(args.no_runs):
    
        gauss_noise = np.random.normal(0, 1, size=manifold_sampler.data.shape)
        plane_noise = np.zeros_like(gauss_noise)
        for d in range(Gv.shape[0]):
            proj = np.dot(gauss_noise, Gv[d])
            for s in range(plane_noise.shape[0]):
                plane_noise[s] = plane_noise[s] + proj[s]*Gv[d]        
        ortho_noise = gauss_noise - plane_noise

        RADIUS = args.sampler_noise

        # noise
        ortho_norm = utils.normalize(ortho_noise, RADIUS)
        plane_norm = utils.normalize(plane_noise, RADIUS)
        gauss_norm = utils.normalize(gauss_noise, RADIUS)

        # point clouds
        ortho_pc = manifold_sampler.data + ortho_norm
        plane_pc = manifold_sampler.data + plane_norm
        gauss_pc = manifold_sampler.data + gauss_norm

        rips = Rips()
        diagrams_in = rips.fit_transform(data)
        diagrams_gauss = rips.fit_transform(gauss_pc)
        diagrams_plane = rips.fit_transform(plane_pc)
        diagrams_ortho = rips.fit_transform(ortho_pc)

        bottleneck_distances = (bottleneck(diagrams_in[0], diagrams_gauss[0]), bottleneck(diagrams_in[1], diagrams_gauss[1]),
                                bottleneck(diagrams_in[0], diagrams_plane[0]), bottleneck(diagrams_in[1], diagrams_plane[1]),
                                bottleneck(diagrams_in[0], diagrams_ortho[0]), bottleneck(diagrams_in[1], diagrams_ortho[1]))

        if args.no_runs == 1:
            utils.viz_for_paper(data, 
                              gauss_pc,
                              plane_pc,
                              ortho_pc,
                              H = bottleneck_distances,
                              name = args.syn_sub)
        else:
            bottleneck_results.append(np.asarray(bottleneck_distances))
    
    filename = 'results/' + args.syn_sub
    with open(filename, 'wb') as outfile:
        pickle.dump(bottleneck_results, outfile)
    
def gen_circle(args):
    print("---Generate circle data")
    n = args.no_points
    r = 1
    phi = np.pi / 6 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 2*np.pi, n)
    zvert = np.linspace(0, 0, n)
    
    xdata = r*np.cos(theta) + noise * np.random.randn(n)
    ydata = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) + zvert
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/circle3d_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name

def gen_spiral(args):
    print("---Generate spiral data")
    n = args.no_points
    phi = np.pi / 6 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 3.2*np.pi, n)
    zvert = np.linspace(0, 1, n)
    r =  np.linspace(0, 2, n)
    
    xdata = r*np.cos(theta) + noise * np.random.randn(n)
    ydata = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) + zvert
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/spiral3d_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name

def gen_spiral2(args):
    print("---Generate spiral2 data")
    n = args.no_points
    phi = np.pi / 6 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 3.2*np.pi, n)
    r = np.linspace(0, 2, n)
    
    xdata = r*np.cos(theta) + noise * np.random.randn(n)
    ydata = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/spiral3d2_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name

def gen_spiral3(args):
    print("---Generate spiral3 data")
    n = args.no_points
    phi = np.pi / 6 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 3.2*np.pi, n)
    r = np.linspace(0, 2, n)
    
    xdata = r*np.cos(theta) + noise * np.random.randn(n)
    ydata = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/spiral3d3_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name

def gen_line(args):
    print("---Generate line data")
    n = args.no_points
    phi = np.pi / 6 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    
    r =  np.linspace(0, 10, n)
    
    xdata = r + noise * np.random.randn(n)
    ydata = r + noise * np.random.randn(n)
    zdata = r + noise * np.random.randn(n)
    
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/line3d_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name

def gen_circle2(args):
    print("---Generate 2 circles data")
    n = args.no_points
    r = 1
    offset = 0.95
    phi = np.pi / 12 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 2*np.pi, n)
    
    xdata = r*np.cos(theta) + noise * np.random.randn(n)
    ydata = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) 
    
    xdata2 = r*np.cos(theta) + noise * np.random.randn(n) + offset
    ydata2 = r*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata2 = r*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) 
    
    xdata = np.concatenate((xdata,xdata2))
    ydata = np.concatenate((ydata,ydata2))
    zdata = np.concatenate((zdata,zdata2))
    
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/circle3d2_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name
    
def gen_circle3(args):
    print("---Generate 2 circles of different sizes")
    n = args.no_points
    r1 = 1
    r2 = 2
    phi = np.pi / 12 # "tilt" of the circle for visuzalization
    noise = args.data_noise
    
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    theta = np.linspace(0, 2*np.pi, n)
    
    xdata = r1*np.cos(theta) + noise * np.random.randn(n)
    ydata = r1*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata = r1*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) 
    
    xdata2 = r2*np.cos(theta) + noise * np.random.randn(n)
    ydata2 = r2*np.sin(theta)*np.cos(phi) + noise * np.random.randn(n)
    zdata2 = r2*np.sin(theta)*np.sin(phi) + noise * np.random.randn(n) 
    
    xdata = np.concatenate((xdata,xdata2))
    ydata = np.concatenate((ydata,ydata2))
    zdata = np.concatenate((zdata,zdata2))
    
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='twilight');
    ax.axis('off')
    fig_name = "viz/circle3d3_" + str(noise) + ".png"
    plt.savefig(fig_name, transparent=True)
    
    return xdata, ydata, zdata, fig_name    