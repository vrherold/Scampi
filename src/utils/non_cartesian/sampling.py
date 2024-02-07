__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2022"
__status__ = "Experimental"

import math
import torch


from utils.plot_utils import plot_traj


def generate_golden_angle_radial_sampling_pattern(spoke_length, num_spokes, dtype=torch.float32):
    theta = torch.zeros(num_spokes, dtype=dtype)
    """ Generate a radial sampling pattern using the golden angle
    #     Args:
    #         spoke_length (int): Number of samples per spoke
    #         num_spokes (int): Number of spokes
    #         dtype (torch.dtype): Data type of the sampling pattern
    #     Returns:
    #     torch.Tensor: Trajectoy (2, spoke_length * num_spokes)
    #     """

    dtheta = 360 / (1 + math.sqrt(5))
    for k in range(1, num_spokes + 1):
        theta[k - 1] = (k - 1) * dtheta
    theta = torch.deg2rad(theta)

    line = torch.linspace(-math.pi, math.pi, spoke_length)

    kx = torch.ger(line, torch.cos(theta))
    ky = torch.ger(line, torch.sin(theta))

    kspace = torch.stack([kx.t().flatten(), ky.t().flatten()]).to(dtype)
    #plot_traj(kspace, 10, spoke_length=spoke_length)

    return kspace
