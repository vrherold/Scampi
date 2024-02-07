__author__ = "Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

from pathlib import Path
from utils.coilmaps import CoilMaps
from utils.cartesian.sampling import Sampling
from utils.cartesian.transforms import cartesian_forward
from utils.data_utils import load_resize_image
import torch


class CartesianKspaceGenerator:
    """
    Generates multi-coil-kspace-data from an image
    """

    def __init__(self, image_path: Path, dims: tuple, n_coilmaps: int = 1, accel_factor: int = 1):
        self.image = None
        self.dims = dims
        self.kspace = None
        self.mask = None
        self.coilmaps = None

        self._load_image(image_path)

        self.coilmaps = CoilMaps(image_path.parent)
        self._create_coilmaps(n_coilmaps)

        if accel_factor > 1:
            self.accel_factor = accel_factor
            self.mask = Sampling(image_path.parent)
            self._create_mask()

    def _load_image(self, image_path: Path):
        self.image = load_resize_image(image_path, self.dims)
        # self.dim = self.image.shape

    def _create_coilmaps(self, n_coilmaps: int):
        self.coilmaps.generate_coilmap_container(1, shape=(n_coilmaps, self.dims[-2], self.dims[-1]), r=1.5)

    def _create_mask(self):
        self.mask.generate_mask_container(1, self.dims, accel=self.accel_factor, rsampling='gaussian', scale=1.0)

    def forward(self, file_name: str = None):
        cm = self.coilmaps.rndraw_maps(1)
        mask = self.mask.rndraw_samplings(1).transpose(-1, -2)
        self.kspace = cartesian_forward(self.image, cm, mask).squeeze(0)
        if file_name is not None:
            torch.save(self.kspace, file_name + '.pt')
        return self.kspace
