__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2022"
__status__ = "Experimental"

import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import mda_slice
import torch
import matplotlib.cm as cmc
import math

def disp(input1, input2=None, dims=(1, 2), scale_error=1.0, log=False):
    # Ensure inputs are numpy arrays
    if torch.is_tensor(input1):
        input1 = input1.cpu().numpy()
    if input2 is not None and torch.is_tensor(input2):
        input2 = input2.cpu().numpy()

    if len(input1.shape) > 2:
        # Select two dimensions for multidimensional arrays
        input1 = mda_slice(input1, dims)
    # Calculate magnitude and phase for input1
    if not log:
        mag1 = np.abs(input1)
    else:
        mag1 = np.log(np.abs(input1) + .002)

    phase1 = np.angle(input1)

    plt.figure(figsize=(10, 10))
    plt.gray()

    if input2 is not None:
        if len(input2.shape) > 2:
            # Select two dimensions for multidimensional arrays
            input2 = mda_slice(input2, dims)

        phase2 = np.angle(input2)

        if not log:
            mag2 = np.abs(input2)
        else:
            mag2 = np.log(np.abs(input2) + .002)

        # Calculate difference
        diff = np.abs(mag1 - mag2)

        # Plotting
        plt.subplot(3, 2, 1)
        plt.imshow(mag1.transpose())
        plt.title('Magnitude of Input1')
        plt.colorbar()

        plt.subplot(3, 2, 2)
        plt.imshow(phase1.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase of Input1')
        plt.colorbar()

        plt.subplot(3, 2, 3)
        plt.imshow(mag2.transpose())
        plt.title('Magnitude of Input2')
        plt.colorbar()

        plt.subplot(3, 2, 4)
        plt.imshow(phase2.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase of Input2')
        plt.colorbar()

        plt.subplot(3, 2, 5)
        plt.imshow(diff.transpose(), vmin=0, vmax=mag1.max() / scale_error)
        plt.title(' Error-Map (scaling : ' + str(scale_error) + ')')
        plt.colorbar()

    else:
        # Plotting
        plt.subplot(2, 1, 1)
        plt.imshow(mag1.transpose())
        plt.title('Magnitude')
        plt.colorbar()

        plt.subplot(2, 1, 2)
        plt.imshow(phase1.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase')
        plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_traj(traj, n_spokes=40, spoke_length=256):
    tr = traj.cpu().numpy()

    # plot the first 40 spokes
    kx = tr[0, :n_spokes * spoke_length]
    kx = np.reshape(kx, (n_spokes, spoke_length))
    ky = traj[1, :n_spokes * spoke_length]
    ky = np.reshape(ky, (n_spokes, spoke_length))

    viridis = cmc.get_cmap('viridis', n_spokes)
    colors = viridis(np.linspace(0, 2, n_spokes))
    for i in range(n_spokes):
        plt.scatter(kx[i, :], ky[i, :], s=3, color=colors[i])

    # plt.scatter(kx[:40, :].transpose(), ky[:40, :].transpose())
    plt.axis('equal')
    plt.title('k-space trajectory (first {n_spokes} spokes)')
    plt.show()
