__author__ = "Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

import torch
from typing import Union, Tuple, List, Iterable
import numpy as np
TensorOrArray = Union[np.ndarray, torch.Tensor]


def multi_dim_fft_torch(input: torch.Tensor, mask: Union[Iterable, Tuple, List] = None,
                        inverse: bool = False) -> torch.Tensor:
    """
    Centered, (inverse) fourier transform.
    Performs a multidimensional centered fourier transform (from image-space to k-space) according to the
    transformation mask.

    Args:
        input: input (torch.tensor).
        mask: (optional) = [1/0 , 1/0, ... for each dimension]. 1: transform along this dimension. 0: keep the domain
        inverse: if True, performs an inverse fourier transform

    Returns:
        result: fourier-domain (torch.tensor)

    """
    if mask.__len__() != input.dim():
        raise ValueError("The number of entries in mask must match the number of image dimensions.")
    axis = [i for i, j in enumerate(mask) if j == 1]
    if inverse:
        res = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(input, dim=axis), dim=axis,
                                                 norm="ortho"), dim=axis)
    else:
        res = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(input, dim=axis),
                                                dim=axis,
                                                norm="ortho"),
                                 dim=axis)

    return res


def image2kspace_torch(image: torch.Tensor, mask: Union[Iterable, Tuple, List] = None) -> torch.Tensor:
    return multi_dim_fft_torch(image, mask=mask, inverse=False)


def kspace2image_torch(kspace: torch.Tensor, mask: Union[Iterable, Tuple, List] = None) -> torch.Tensor:
    return multi_dim_fft_torch(kspace, mask=mask, inverse=True)


def cartesian_backward(ksp: torch.Tensor, cm: torch.Tensor) -> torch.Tensor:
    """
    Backward operator for cartesian sampling.

    Args:
        ksp : k-space (n_coils, nx, ny)
        cm : Coilmaps  (n_coils, nx, ny)

    Returns:
        img: Image space tensor of shape (nx, ny)

    """

    assert ksp.shape == cm.shape, f'Dimensional mismatch between k-space and Sensitivity Data!'

    img_array_c = kspace2image_torch(ksp, [0, 0, 1, 1])

    norm = torch.sum(torch.square(torch.abs(cm)), dim=1)
    norm = torch.where(torch.Tensor(norm == 0), torch.finfo(cm.dtype).eps, norm)  # avoid division by zero
    res = torch.sum(img_array_c * torch.conj(cm), dim=1) / norm

    return res


def cartesian_forward(img: torch.Tensor, cm: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Backward operator for cartesian sampling.

    Args:
        img : image (n_coils, nx, ny)
        cm : Coilmap of shape (n_coils, nx, ny)
        mask: Sampling mask of shape (nx, ny)

    Returns:
        (us)kspace: undersampled kspace (n_coils, nx, ny)

    """
    image_maps = cm * img.broadcast_to(cm.shape)
    ksp = image2kspace_torch(image_maps, (0, 0, 1, 1))

    if mask is not None:
        ksp = ksp * mask.broadcast_to(ksp.shape)
    return ksp
