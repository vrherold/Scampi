__author__ = "Thomas Siedler, Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

import random

import scipy
import torch
import math as m
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from sigpy.mri.samp import poisson
from utils.data_utils import rndraw
import glob
import os


class Sampling:
    """Wrapper class for sampling patterns."""

    def __init__(self, filename: Union[Path, str] = './sampling'):
        """Initialize Sampling object.

        On initialization, the object only contains the path to the Numpy array, that stores the sampling pattern.
        The masks themselves are only loaded right before being passed to the forward operator.
        """

        self._filename = filename

    def generate_poisson_disk(self, img_shape: Tuple[int, ...], accel: float, K: float = 30, calib: Tuple[int] = (0, 0),
                              dtype=np.complex, crop_corner: bool = True, return_density: bool = False,
                              seed: int = 0) -> None:
        """Call sigpy.mri.samp.poisson and save return value to numpy array."""

        if return_density:
            mask, density = poisson(img_shape, accel, K, calib, dtype, crop_corner, return_density, seed)
            np.save(self._filename, mask)
            np.save(self._filename[:-4] + '_density.npy', density)
        else:
            mask = poisson(img_shape, accel, K, calib, dtype, crop_corner, return_density, seed)
            # np.save(self._filename, mask)
            mask_t = torch.from_numpy(mask)
            torch.save(mask_t, self._filename)

    def generate_2d_mask(self, img_shape: Tuple[int, ...], accel: float, rsampling: str, save: bool = True,
                         **kwargs) -> Union[None, np.ndarray]:
        """Generate a subsampling mask for 2dimensional data (for example rows and columns of 2d kspace data).

        Parameters
        ----------
        img_shape : input-shape of k-space data. Only 2d accepted.

        accel :  Acceleration factor
                    ->  N_data = int(1/accel) * lines
                        with lines = img_shape[1]

        rsampling :  Method of sampling
                    - 'uniform':    Each line has the same probability to be sampled
                    - 'gaussian':   The probability of a line to be sampled follows a Gaussian distribution centered
                    around the median line. The central n lines will always be sampled. n can be given through
                    keyword 'calibregion'. The default is 10% of total lines. The 'gaussian' option also requires
                    keywords 'scale' to define the sigma factor of the Gauss curve.

        save : Save the resulting mask in `self.filename`. Is set false, when called out of
               `self.create_mask_container`.

        kwargs :
          - scale :  Additional parameter for the probability distribution, where required. E.g. for 'gaussian',
                     scale sets the sigma factor of the Gaussian curve.
          - calibregion :  int to how many central lines are always to be sampled.

        Returns
        -------
         np.ndarray: A 2D subsampling mask (np.ndarray of img_shape and dtype=bool)

        """

        if isinstance(img_shape, int):
            img_shape = (img_shape, img_shape)  # turn img_shape into tuple
        elif isinstance(img_shape, tuple) and len(img_shape) == 2:
            pass
        else:
            ValueError('The img_shape passed to generate_2d_mask cannot be interpreted!')

        cols, lins = img_shape[0], img_shape[1]

        mask_y = None
        mask = None
        if accel > 1:
            if rsampling == 'uniform':
                mask_y = np.full((img_shape[1]), False)
                n_el = int(img_shape[1] / accel)
                indices = random.sample(range(0, img_shape[1]), n_el)
                mask_y[indices] = True
                flag_linepattern = True

            elif rsampling == 'gaussian':
                scale = kwargs.get('scale', None)
                calibregion = kwargs.get('calibregion', int(lins / 10))
                mask_y = np.full((img_shape[1]), False)
                indices = self._gaussian_like(lines=lins, calibregion=calibregion, accel=accel, scale=scale)
                mask_y[indices] = True
                flag_linepattern = True

            elif rsampling == 'centersquare':
                calibregion = kwargs.get('calibregion', int(lins / 10))
                mask_y = np.full((img_shape[1]), False)
                middle = int(lins / 2)
                ce_l, ce_r = m.floor(calibregion / 2), m.ceil(calibregion / 2)  # handle odd calibregion
                center = range(middle - ce_l, middle + ce_r)
                mask_y[center] = True
                flag_linepattern = True

            elif rsampling == 'poisson_disk':
                mask = poisson(img_shape, accel, return_density=False)
                flag_linepattern = False

            elif rsampling == 'regular':
                assert isinstance(accel, int) or (isinstance(accel, float) and accel.is_integer()), "regular sampling" \
                                                                                                    "works only with" \
                                                                                                    "integer R factor"
                mask_y = np.full((img_shape[1]), False)
                calibregion = kwargs.get('calibregion', None)
                if calibregion is not None:
                    middle = int(lins / 2)
                    ce_l, ce_r = m.floor(calibregion / 2), m.ceil(calibregion / 2)  # handle odd calibregion
                    center = range(middle - ce_l, middle + ce_r)
                    mask_y[center] = True
                mask_y[::accel] = True
                flag_linepattern = True

            else:
                raise KeyError("Cannot interpret the given sampling pattern.")

            if flag_linepattern:
                if img_shape[0] == 1:
                    mask = mask_y[np.newaxis, :]
                else:
                    mask = np.repeat(mask_y[:, np.newaxis], img_shape[0], axis=1)

            if not flag_linepattern:
                pass

        else:
            ValueError('Error! acceleration given to generate_2d_mask() is not  > 1 ')

        if save:
            # np.save(self._filename, mask)
            mask_t = torch.from_numpy(mask)
            torch.save(mask_t, self._filename)
        else:
            return mask

    def _gaussian_like(self, lines: int, calibregion: int, accel: float = 2.0, scale=None) -> np.ndarray:
        """Randomly select integers between 0 and `lines`.

        Draw integers from a range from 0 to `lines` (without replacement). The number n of drawn integers is
        determined by n=`lines`/accel. The probability for each integer to be picked is determined by a normal
        distribution normalized to the limited intervall [0,`lines`]. Because of that normalization and the forced
        calibration region the distribution is only gaussian-'like'.
        Parameters:
        :param lines:       Upper limit of the range to pick integers from, i.e. the function returns randomly drawn
                            integers between 0 and `lines`.
        :param calibregion: the n=`calibregion` integers in the center of the intervall [0, `lines`] will always be
                            selected.
        :param accel:       The acceleration determines the fraction 1/acceleration of values to pick from [0,`lines`].
        :param scale:       Scale factor for the sigma-factor of the normal distribution.

        """
        if scale is None:
            raise KeyError('No scale for gaussian specified!')

        b = scipy.stats.norm.pdf(range(lines), loc=float(lines / 2), scale=scale * lines)
        b = b / np.sum(b)
        pick_idx = np.random.choice(range(lines), size=int(lines / accel), replace=False, p=b)

        if calibregion > 0:
            middle = int(lines / 2)
            ce_l, ce_r = m.floor(calibregion / 2), m.ceil(calibregion / 2)  # handle odd calibregion
            center = range(middle - ce_l, middle + ce_r)
            if len(center) >= int(lines / accel):
                raise ValueError('Selected calibregion is larger than number of lines for selected acceleration!')
            pick_idx = self._fill_center(pick_idx, center)

        return pick_idx

    def generate_mask_container(self, n_masks: int, img_shape: Tuple[int], accel: Union[float, tuple], rsampling: str,
                                **kwargs) -> None:
        """
        Generate a set of n masks and save them as torch tensors at self._filename.

        Parameters
        ----------
        n_masks : Number of different sampling patterns to generate. During training/testing a random sample out of the n patterns will be drawn.
        img_shape : Shape of one mask. A tuple of integers in form (pxl_x, pxl_y) is expected.
        accel : Acceleration factor. Note that real acceleration factor might differ, if options like 'calibregion' to always sample the center are passed.
        rsampling : 'gaussian', 'centersquare' or 'uniform'
        kwargs : See Sampling.generate_2d_mask()

        Returns
        -------
        None

        """

        if isinstance(accel, tuple):
            try:
                accelrange = [i for i in np.arange(*accel)]
                flag_accelrange = True
            except TypeError:
                print('accel should be passed as float or tuple of floats!')
        elif isinstance(accel, (float, int)):
            flag_accelrange = False
            accel_i = accel

        if not os.path.exists(str(self._filename) + '/sampling/'):
            # If not, create the directory
            os.makedirs(str(self._filename) + '/sampling/')

        for i in range(n_masks):
            str(i).zfill(5)
            fn = str(self._filename) + '/sampling/' + 'sm_' + str(i).zfill(5) + '.pt'
            if flag_accelrange:
                accel_i = random.choice(accelrange)
            mask = torch.from_numpy(
                self.generate_2d_mask(img_shape=img_shape, accel=accel_i, rsampling=rsampling, save=False, **kwargs))
            torch.save(mask, fn)

    @staticmethod
    def _fill_center(indexarray: np.ndarray, center: range) -> np.ndarray:
        """Manipulate the random seletion of random sampling patterns like gaussian_like to fill the calibregion. """
        j = 0
        for i in center:
            if i in indexarray:
                pass  # Do not replace another index from the center. Pass and try another one.
            else:
                while True:
                    if indexarray[j] in center:
                        j += 1
                        continue
                    if indexarray[j] not in center:
                        indexarray[j] = i
                        j += 1
                        break
        return indexarray

    def rndraw_samplings(self, k: int = 1):
        """Draw randomly a sampling pattern from the coilmap folder."""
        filelist = glob.glob(self._filename.__str__() + '/sampling/' + '/*.pt')
        return rndraw(filelist, k)

    def load(self):
        """Load and return torch tensor with sampling pattern."""
        return torch.load(self._filename)


def data_consistency(k, k0, mask, noise_lvl=None):
    """Replaces reconstructed lines with actually measured lines, where possible.


    :param k: input in k-space
    :param k0: initially sampled elements in k-space
    :param mask: 1 corresponding nonzero location (measured), 0 corresponding non-measured location
    :param noise_lvl: (optional) Float indicating noise level of measurement, allowing also to replace actually measured
    lines.
    :return: Data-consistent reconstruction of k.
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out
