__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2022"
__status__ = "Experimental"


import torch
from typing import Union
import numpy as np
import torchkbnufft as tkbn
TensorOrArray = Union[np.ndarray, torch.Tensor]


def radial_backward(kdata: torch.Tensor = None, im_size: tuple = None, coilmaps: torch.Tensor =None, ktraj:torch.Tensor= None,
                    nufft_ob_adj:tkbn.KbNufftAdjoint= None) -> torch.Tensor:
    """
    Backward operator for radial sampling.

    Parameters
    ----------
    kdata : k-space (n_coils, nx, ny)
    im_size : final Image size (nx, ny)
    coilmaps : Coilmaps  (n_coils, nx, ny)
    ktraj : k-space trajectory
    nufft_ob_adj : adjoint NUFFT object

    Returns
    -------
    img: Image space tensor of shape (nx, ny)

    """
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    img_array_c = nufft_ob_adj(kdata * dcomp, ktraj)

    norm = torch.sum(torch.square(torch.abs(coilmaps)), dim=1)
    norm = torch.where(torch.Tensor(norm == 0), torch.finfo(coilmaps.dtype).eps, norm)  # avoid division by zero
    res = torch.sum(img_array_c * torch.conj(coilmaps), dim=1) / norm

    return res


def radial_forward(image: torch.Tensor = None, coilmaps: torch.Tensor = None, ktraj:torch.Tensor =None, nufft_ob:tkbn.KbNufft=None,
                  noise: bool = False):
    """
        Forward operator for radial sampling.

        Parameters
        ----------
        image : image (n_coils, nx, ny)
        coilmaps : Coilmap of shape (n_coils, nx, ny)
        ktraj : k-space trajectory
        nufft_ob : NUFFT object
        noise : Add noise to the kspace data

        Returns
        -------
        (us)kspace: undersampled kspace (n_coils, nx, ny)

        """
    if coilmaps is not None:
        images = coilmaps * torch.broadcast_tensors(image.unsqueeze(0), coilmaps)[0]
        kdata = nufft_ob(images, ktraj)
    else:
        images = image
        kdata = nufft_ob(images.unsqueeze(0), ktraj)


    if noise:
        # add some noise (robustness test)
        siglevel = torch.abs(kdata).mean()
        kdata = kdata + (siglevel / 5) * torch.randn(kdata.shape).to(kdata)
    return kdata
