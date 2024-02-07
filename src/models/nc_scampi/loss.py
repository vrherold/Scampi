__author__ = "Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

# Imports

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss  # , L1Loss, MSELoss
from utils.data_utils import toComplex, toReal  # , sobel_operator


class L1L2Loss(_Loss):
    def __init__(self, eta1, eta2, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(L1L2Loss, self).__init__(size_average, reduce, reduction)
        self.eta1 = eta1
        self.eta2 = eta2

    # noinspection PyShadowingBuiltins
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        loss = self.eta1 * F.l1_loss(input_tensor, target_tensor, reduction=self.reduction) \
               + self.eta2 * F.mse_loss(input_tensor, target_tensor, reduction=self.reduction)
        return loss / (self.eta1 + self.eta2)


class NCScampiLoss(_Loss):
    """
    NUFFT-based loss for non-cartesian MRI reconstruction.

    """

    def __init__(self, nufft_forward=None, nufft_backward=None, reduction: str = 'mean', device=None) -> None:
        super(NCScampiLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)

        self.nufft_forward = nufft_forward
        self.nufft_backward = nufft_backward

        self.device = device

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dtype not in (
                torch.complex64, torch.complex128):  # input_tensor is not complex during training (stacked complex)
            input_tensor = toComplex(input_tensor, 1)

        # sobel_output = sobel_operator(input_tensor)
        # sharpness_loss = torch.mean(sobel_output)
        # n_coils = target_tensor.shape[1] / 2

        # loss1 = F.mse_loss(toReal(self.nufft_forward(input_tensor.squeeze(0)), 1), target_tensor,
        #                   reduction=self.reduction).float()

        loss2 = F.l1_loss(toReal(self.nufft_forward(input_tensor.squeeze(0)), 1), target_tensor,
                          reduction=self.reduction).float()

        loss = loss2

        return loss


def l1_wavelets(coeffs_real: tuple, coeffs_imag: tuple) -> torch.Tensor:
    a = torch.sum(
        torch.abs(torch.view_as_complex(torch.stack((coeffs_real[0], coeffs_imag[0]), dim=4))))
    b = torch.stack(
        [torch.abs(torch.view_as_complex(torch.stack((tr, ti
                                                      ), dim=5))).sum() for tr, ti in
         zip(coeffs_real[1], coeffs_imag[1])]).sum()
    return a + b
