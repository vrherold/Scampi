__author__ = "Volker Herold"
__year__ = "2023"
__status__ = "Experimental"

from pathlib import Path
import torch
from utils.coilmaps import CoilMaps
from utils.data_utils import load_resize_image
from utils.non_cartesian.transforms import radial_forward, radial_backward
from utils.params import dtype_cmapping

import torchkbnufft as tkbn


class NonCartesianKspaceGenerator:  # default is radial
    def __init__(self, im_size: tuple = None, grid_size: tuple = None, func_traj=None,
                 kwargs_traj: dict = None, device=torch.device('cpu'), dtype: torch.dtype = None):

        self.dtype = dtype
        self.dtype_c = dtype_cmapping[dtype.__str__().rsplit('.')[
            1]]  # extracts the 'float...' as a string from dtype and maps it to the complex type
        self.image = None
        self.im_size = im_size
        if grid_size is None:
            self.grid_size = (2 * im_size[0], 2 * im_size[1])
        else:
            self.grid_size = grid_size

        self.device = device
        self.nufft_op = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, device=device)
        self.nufft_op_adj = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, device=device)
        self.kdata = None
        self.coilmaps = None  # container for coilmaps
        self.cms = None  # coilmaps
        self.traj_generator = func_traj
        self.kwargs_traj = kwargs_traj
        self.ktraj = None

    def _load_image(self, image: Path):
        self.image = load_resize_image(image, self.im_size).to(self.device).to(self.dtype_c)

    def _create_coilmaps(self, n_coilmaps: int):
        self.coilmaps.generate_coilmap_container(1, shape=(n_coilmaps, self.im_size[-2], self.im_size[-1]), r=1.5)
        self.cms = self.coilmaps.rndraw_maps(1).to(self.device).to(self.dtype_c)

    def _create_ktraj(self):
        if self.traj_generator is None or self.kwargs_traj is None:
            raise ValueError("Set Trajectory Generator first with 'set_traj_generator' !")
        self.ktraj = self.traj_generator(**self.kwargs_traj).to(self.dtype).to(self.device).unsqueeze(0)

    def forward(self, image_path: Path = None, n_coilmaps: int = 1):
        self._create_ktraj()
        self.coilmaps = CoilMaps(image_path.parent)
        self._create_coilmaps(n_coilmaps)
        self._load_image(image_path)

        # create k-space data
        self.kdata = radial_forward(self.image, self.cms, self.ktraj, self.nufft_op, noise=False)
        return self.kdata

    def set_traj_generator(self, func=None, **kwargs):
        self.kwargs_traj = kwargs
        self.traj_generator = func

    def backward(self):
        if self.ktraj is None:
            raise ValueError("No k-space trajectory and/or kspace-data available. Run forward() first.")
        else:
            return radial_backward(kdata=self.kdata, im_size=self.im_size, coilmaps=self.cms, ktraj=self.ktraj,
                                   nufft_ob_adj=self.nufft_op_adj)

    def get_trajectory(self):
        if self.ktraj is None:
            raise ValueError("Calculate Trajectory first !")
        else:
            return self.ktraj

    def get_coilmaps(self):
        if self.cms is None:
            raise ValueError("No coilmaps available: Singe Coil Mode !")
        return self.cms

    def get_nufft_op(self):
        return self.nufft_op

    def get_nufft_op_adj(self):
        return self.nufft_op_adj

    def get_image(self):
        if self.image is None:
            raise ValueError("No image available yet. Run forward() first !")
        else:
            return self.image
