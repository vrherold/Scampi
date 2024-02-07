__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2022"
__status__ = "Experimental"

from sigpy.mri.sim import birdcage_maps
import random
import torch
import numpy as np
from typing import Union, Tuple
from pathlib import Path
import glob
import os
from utils.data_utils import rndraw


class CoilMaps:
    """ Wrapper class for Coilmaps."""

    def __init__(self, filename: Path = ''):
        """Initialize CoilMaps object.

        On initialization, the object only contains the path to the Numpy array, that stores the coilmaps. The data
        themselves are only loaded right before being passed to the forward operator in LabelListMRI.__getitem__ method.
        """

        self._filename = filename
        self.n_coils = None  # not set yet see generate...

    def generate_birdcage_map(self, shape: Tuple[int], r: float = 1.5, nzz: int = 8, dtype=np.cdouble) -> None:
        """ Generate Coilmap and save it as torch tensor.

        Parameters
        ----------
        shape : Sensitivity maps' shape, can be of length 3 (Nc, Nx, Ny) or 4 (Nc, Nx, Ny, Nz).
        r : relative radius of birdcage.
        nzz : number of coils per ring along z direction
        dtype :  dtype of the saved numpy array.

        Returns
        -------
        None
        """

        res = birdcage_maps(shape, r, nzz, dtype)
        res = torch.from_numpy(res)
        self.n_coils = shape[0]
        torch.save(res, self._filename)

    def generate_coilmap_container(self, n_coilmaps: int, shape: Tuple, r: Union[Tuple[float], float],
                                   nzz: int = 8, dtype=np.cdouble) -> None:
        """
        Generate a set of n coilmaps and save them at self._filename.

        Parameters
        ----------
        n_coilmaps : Number of different coilmaps to generate. During training/testing a random
        sample out of the n coilmaps will be drawn.
        shape : Shape of one coilmap. A tuple of integers in form (n_coils, pxl_x, pxl_y, [pxl_z]) is expected.
        r : Radius/penetration depth of the coil
        nzz : Number of coils along z direction (only relevant for 4-dim shape tuple)
        dtype : Datatype of coilmap container.

        Returns
        -------
        None
        """

        if isinstance(r, tuple):
            r_range = [i for i in np.arange(*r)]
            flag_r_range = True
            r_i = None
        elif isinstance(r, (float, int)):
            r_range = None
            r_i = r
            flag_r_range = False
        else:
            raise ValueError

        if not os.path.exists(str(self._filename) + '/coilmaps/'):
            # If not, create the directory
            os.makedirs(str(self._filename) + '/coilmaps/')

        for i in range(n_coilmaps):
            str(i).zfill(5)
            fn = str(self._filename) + '/coilmaps/cm_' + str(i).zfill(5) + '.pt'
            if flag_r_range:
                r_i = random.choice(r_range)
            torch.save(torch.from_numpy(birdcage_maps(shape, r_i, nzz, dtype)), fn)
        self.n_coils = shape[0]

    def rndraw_maps(self, k: int = 1):
        """Draw randomly k maps from the coilmap folder."""
        filelist = glob.glob(self._filename.__str__() + '/coilmaps' + '/*.pt')
        return rndraw(filelist, k)
