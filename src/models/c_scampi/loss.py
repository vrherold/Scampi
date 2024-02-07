__author__ = "Volker Herold, Thomas Siedler"
__year__ = "2023"
__version__ = "0.0.1"

# Imports

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss
from pytorch_wavelets import DWTForward
from utils.data_utils import toComplex, toReal, fdiff
from utils.cartesian.transforms import kspace2image_torch, image2kspace_torch
from functools import partial


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


class MultiDomainLoss(_Loss):
    def __init__(self, eta_k: float, eta_img: float,
                 input_domain: str = 'kspace', k_loss: _Loss = L1Loss(), img_loss: _Loss = MSELoss(),
                 real_repr=True, n_coils: int = None,
                 size_average: bool = None, reduce: bool = None, reduction: str = 'mean') -> None:
        super(MultiDomainLoss, self).__init__(size_average, reduce, reduction)
        self.eta_k = eta_k
        self.eta_img = eta_img

        if input_domain in ('kspace', 'k'):
            self.input_domain = 'k'
            self.ft = kspace2image_torch
        elif input_domain in ('imgspace', 'image', 'img'):
            self.input_domain = 'img'
            self.ft = image2kspace_torch
        else:
            raise KeyError(f"{self.__name__} got unknown input domain {self.input_domain}.")

        self.k_loss = k_loss
        self.img_loss = img_loss
        for loss in (self.k_loss, self.img_loss):
            if not isinstance(loss, _Loss):
                raise Warning(f"Passed Loss {loss} of type {type(loss)}, but MultiDomainLoss expects loss of"
                              f"type _Loss (or inherited). Maybe {loss}.__init__() was not called?")
        self.real_repr = real_repr
        self.n_coils = n_coils
        self.coil_idx = 1
        self.ft_mask = [0, 0, 1, 1]

    def __repr__(self):
        rep = f'MDL: ' \
              f'eta_k: {self.eta_k} - {type(self.k_loss).__name__}; ' \
              f'eta_img: {self.eta_img} - {type(self.img_loss).__name__}'
        return rep

    def _get_ft(self, t: torch.Tensor) -> torch.Tensor:

        res = self.ft(toComplex(t, dim=1), self.ft_mask)
        res = toReal(res, dim=1)
        return res

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        (in_ft, tar_ft) = tuple([self._get_ft(t) for t in (input_tensor, target_tensor)])

        if self.input_domain == 'k':
            k_loss = self.k_loss(input_tensor, target_tensor) if self.eta_k != 0 else 0
            img_loss = self.img_loss(in_ft, tar_ft) if self.eta_img != 0 else 0
        else:
            img_loss = self.img_loss(input_tensor, target_tensor) if self.eta_img != 0 else 0
            k_loss = self.k_loss(in_ft, tar_ft) if self.eta_k != 0 else 0

        return (self.eta_k * k_loss + self.eta_img * img_loss) / (self.eta_k + self.eta_img)


class ScampiLoss(_Loss):
    r"""Loss with multiple terms to enforce different features of the output image.

    Expects input in k-Space and calculates the multi-term loss function

    .. math::
        \mathcal{L}(x, k_0)
    .. math::
         = \eta_k \,  \mathrm{Loss}_k [PFx - k_0 ]
    .. math::
        + \eta_{img} \,  \mathrm{Loss}_{img} [ F^{-1}{PF}x - F^{-1}k_0 ]
    .. math::
        + l_{L1W} \, \mathrm{L1W}[F^{-1}{DC}(Fx, k_0)]
    .. math::
        + l_{TV} \, \mathrm{TV}[F^{-1}{DC}(Fx, k_0)]

    with different weighting factors :math:`\eta_k, \eta_{img}, l_{L1W}, l_{TV}`. Here, L1W is L1 Wavelet sparsity and
    TV is Total Variation sparsity norm.
    """

    def __init__(self, sampling_mask: torch.Tensor, coilmap: torch.Tensor = None, eta_k: float = 20,
                 eta_img: float = 10,
                 l_l1w: float = 2.5e-8, l_tv=0.0, k_loss: _Loss = L1Loss(), img_loss: _Loss = MSELoss(),
                 reduction: str = 'mean', device: str = 'cpu') -> None:
        r"""Loss with multiple terms to enforce different features of the output image.

        Expects input in k-Space and calculates the multi-term loss function

        .. math::
            \mathcal{L}(x, k_0)
        .. math::
             = \eta_k \,  \mathrm{Loss}_k [PFx - k_0 ]
        .. math::
            + \eta_{img} \,  \mathrm{Loss}_{img} [ F^{-1}{PF}x - F^{-1}k_0 ]
        .. math::
            + l_{L1W} \, \mathrm{L1W}[F^{-1}{DC}(Fx, k_0)]
        .. math::
            + l_{TV} \, \mathrm{TV}[F^{-1}{DC}(Fx, k_0)]

        with different weighting factors :math:`\eta_k, \eta_{img}, l_{L1W}, l_{TV}`.
        Here, L1W is L1 Wavelet sparsity and TV is Total Variation sparsity norm.

        :param sampling_mask: Sampling pattern of k0, required for Data-Consistency operator
        :param l_l1w: Weighting for L1 Wavelet Sparsity.
        :param l_tv: Weighting for Total Variation Sparsity.
        :param k_loss: Loss function in k-space.
        :param img_loss: Loss function in image space.
        :param eta_k: Weighting for loss in k-space.
        :param eta_img: Weighting for loss in image space.
        :param reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super(ScampiLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)

        self.mdl = MultiDomainLoss(eta_k=eta_k, eta_img=eta_img, input_domain='k', k_loss=k_loss, img_loss=img_loss)

        self.l_tv = l_tv
        self.l_l1w = l_l1w

        self.coil_idx = 1
        self.fft_mask = (0, 0, 1, 1)

        self.sampling = sampling_mask
        self.coilmap = coilmap
        self.op_size = list(sampling_mask.squeeze().shape)
        self.device = device

        assert 1 not in self.op_size, "Size 1 dimension in size. This causes problems in sigpy Wavelet! " \
                                      "Squeezing out ones is recommended!"

        self.op_size[0] = int(list(self.op_size)[0] / 2)
        self.op_size = tuple(self.op_size)

        self.diff_op = partial(fdiff, axes=[-2, -1])

        if 'cuda' in self.device.type:
            self.wop = DWTForward(J=5, wave='db2', mode='zero').cuda()
        else:
            self.wop = DWTForward(J=5, wave='db2', mode='zero')

    def __repr__(self):
        """Print out relevant values, when Loss is stored to Json dict, for example."""
        rep = f'MultiTermLoss: eta_k: {self.mdl.eta_k} - {type(self.mdl.k_loss).__name__}; ' \
              f'eta_img: {self.mdl.eta_img} - {type(self.mdl.img_loss).__name__}; ' \
              f'l_l1w: {self.l_l1w}; lamda: {self.lamda}'
        return rep

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:

        input_tensor = toComplex(input_tensor, dim=1)
        if self.coilmap is not None:
            input_tensor = self.coilmap * input_tensor.broadcast_to(self.coilmap.shape)
        input_tensor = image2kspace_torch(input_tensor, [0, 0, 1, 1])
        input_tensor = toReal(input_tensor, dim=1)

        # MultiDomainLoss expects real-valued inputs
        loss = self.mdl.forward(input_tensor * self.sampling, target_tensor)

        # use the full k-space information before sparsity constraining
        add_lines = torch.where(self.sampling, target_tensor, input_tensor)

        # wop and tv expect complex input
        input_img = kspace2image_torch(toComplex(add_lines, dim=1), self.fft_mask).squeeze(dim=0)

        if self.l_l1w != 0:
            coeff_re = self.wop(torch.real(torch.unsqueeze(input_img, 0)))
            coeff_im = self.wop(torch.imag(torch.unsqueeze(input_img, 0)))
            loss += self.l_l1w * l1_wavelets(coeff_re, coeff_im)

        if self.l_tv != 0:
            total_var = torch.sum(self.diff_op(input_img).abs())
            loss += self.l_tv * total_var

        return loss


def l1_wavelets(coeffs_real: tuple, coeffs_imag: tuple) -> torch.Tensor:
    a = torch.sum(
        torch.abs(torch.view_as_complex(torch.stack((coeffs_real[0], coeffs_imag[0]), dim=4))))
    b = torch.stack(
        [torch.abs(torch.view_as_complex(torch.stack((tr, ti
                                                      ), dim=5))).sum() for tr, ti in
         zip(coeffs_real[1], coeffs_imag[1])]).sum()
    return a + b
