__author__ = "Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

import torch
import torch.nn as nn
from pathlib import Path
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
import ast

from models.c_scampi.loss import ScampiLoss as CScampiLoss
from models.nc_scampi.loss import NCScampiLoss
from utils.data_utils import load_tensor
from sigpy.mri.app import EspiritCalib
import models.c_scampi.unet_model as CScampiModel
import models.nc_scampi.unet_model as NCScampiModel
from functools import partial

from utils.data_utils import toReal, toComplex, mda_slice, DipDataset, Trainer
from utils.cartesian.transforms import cartesian_backward
from utils.non_cartesian.transforms import radial_backward, radial_forward
from utils.params import RecoParams
from utils.params import dtype_mapping, dtype_cmapping
from utils.non_cartesian.generate_kspace import NonCartesianKspaceGenerator
from utils.non_cartesian.sampling import generate_golden_angle_radial_sampling_pattern


class UcnnReco(nn.Module):
    """Base class for all reconstruction methods."""

    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        self.target = None
        self.x = None
        self.measurement_forward = None
        self.measurement_backward = None
        self.dim = None
        self.device = None
        self.n_channels = 2
        self.coilmap = None
        self.dl = None
        self.trainer = None
        self.parameters = None
        self.dtype = torch.float32
        self.dtype_c = torch.complex64
        self.gt = None

    def prep_data(self):
        raise NotImplementedError

    def prep_model(self):
        raise NotImplementedError


class CartesianScampi(UcnnReco):
    """Class for Cartesian reconstruction with SCAMPI."""

    def __init__(self, recopars: RecoParams = None, device='cpu'):
        super().__init__()
        self.device = device
        self.recopars = recopars
        if self.recopars['dtype'] in ['float16', 'float32', 'float64']:
            self.dtype = dtype_mapping[recopars['dtype']]
            self.dtype_c = dtype_cmapping[recopars['dtype']]
        else:
            raise ValueError("dtype in RecoParams must be either float32 or float64")
        self.recopars = recopars
        self.data = {
            'full_kspace': None,
            'us_kspace': None,
            'coilmap': None,
            'mask': None
        }
        self.sampling_mask = None

    def set_data_path(self, full_kspace_path: Path = None, us_kspace_path: Path = None, mask_path: Path = None,
                      coilmap_path: Path = None):
        self.data['full_kspace'] = full_kspace_path
        self.data['us_kspace'] = us_kspace_path
        self.data['mask'] = mask_path
        self.data['coilmap'] = coilmap_path

    def prep_data(self):

        def estimate_coilmap(x):

            print("Estimating CoilMaps...:")
            cm = EspiritCalib(x.numpy()).run()
            cm = torch.from_numpy(cm).unsqueeze(0).to(self.device).to(self.dtype_c)
            print("Done")
            return cm

        # Case 1: full kspace and mask is given (full_kspace, mask)
        if (self.data['full_kspace'] is not None) and (self.data['mask'] is not None):

            print("Fully sampled kspace and mask is given: undersampled "
                  "kspace data will be generated i.e. provided data for us_kspace will be ignored.")

            ksp_full = load_tensor(self.data['full_kspace'])
            self.dim = ksp_full.shape  # (nc, nx, ny)

            self.sampling_mask = load_tensor(self.data['mask'])

            if tuple(self.dim[-2:]) != self.sampling_mask.shape:
                raise ValueError("Shape of kspace and mask must be the same")

            self.target = ksp_full * self.sampling_mask  # broadcasting

            if self.data['coilmap'] is not None:
                self.coilmap = load_tensor(self.data['coilmap']).to(self.device).to(self.dtype_c).unsqueeze(0)
            else:
                self.coilmap = estimate_coilmap(self.target)

            self.gt = cartesian_backward(ksp_full.to(self.device).unsqueeze(0), self.coilmap).to(self.dtype_c)

        # Case 2: only undersampled kspace is given (us_kspace)
        else:
            self.target = load_tensor(self.data['us_kspace'])
            self.dim = self.target.shape  # (nc, nx, ny)
            self.sampling_mask = self.target != 0
            self.coilmap = estimate_coilmap(self.target)

        self.target = toReal(self.target, dim=0).to(device=self.device).unsqueeze(0).to(
            self.dtype)  # real valued with batch dimension
        self.sampling_mask = mda_slice(self.sampling_mask,
                                       (-2, -1))  # only take one replicat of the mask to enable broadcasting

        self.sampling_mask = self.sampling_mask.to(self.device).broadcast_to(self.target.shape)

        self.x = torch.rand((self.n_channels, self.dim[1], self.dim[2])).to(device=self.device).unsqueeze(0).to(
            self.dtype)  # randomly generated input
        self.dl = DipDataset(self.x, self.target)

    def prep_model(self):
        self.model = CScampiModel.DipUnet(self.n_channels, self.n_channels, self.sampling_mask, produce=False,
                                          apply_data_consistency=True, k0=self.target,
                                          coilmap=self.coilmap, skip_connections=self.recopars['skip_connections']).to(
            device=torch.device(self.device)).to(self.dtype)

        loss_func = CScampiLoss(sampling_mask=self.sampling_mask, coilmap=self.coilmap,
                                eta_k=self.recopars['eta_k'], eta_img=self.recopars['eta_img'],
                                l_l1w=self.recopars['l_l1w'], l_tv=self.recopars['l_tv'],
                                k_loss=L1Loss(), img_loss=MSELoss(), device=self.device)

        optimizer = Adam(params=self.model.parameters(), lr=self.recopars['learningrate'])

        trainloader = DataLoader(self.dl)

        self.trainer = Trainer(trainloader, self.model, optimizer, loss_func)

    def forward(self, preview: int = 0):
        self.trainer.train(self.recopars['n_epochs'], name=self.recopars['name'], preview=preview)

        return self.inference()

    def inference(self):
        with torch.no_grad():
            self.model.produce = True
            res = self.model.forward(self.x)

        return res

    def get_gt(self):
        if self.gt is None:
            raise ValueError("Ground truth is not available !")
        else:
            return self.gt


class NonCartesianScampi(UcnnReco):
    """Class for Non-Cartesian reconstruction with SCAMPI."""

    def __init__(self, recopars: RecoParams = None, device='cpu'):
        super().__init__()
        self.device = device
        self.recopars = recopars
        if self.recopars['dtype'] in ['float16', 'float32', 'float64']:
            self.dtype = dtype_mapping[recopars['dtype']]
            self.dtype_c = dtype_cmapping[recopars['dtype']]
        else:
            raise ValueError("dtype in RecoParams must be either float32 or float64")
        self.recopars = recopars
        self.data = {
            'kdata': None,
            'coilmap': None,
            'ktraj': None,
            'image': None
        }
        self.ktraj = None
        self.kdata = None
        self.nuff_op = None
        self.nuff_op_adj = None
        self.im_size = ast.literal_eval(self.recopars['image_size'])
        self.grid_size = ast.literal_eval(self.recopars['grid_size'])

        self.kspace_generator = NonCartesianKspaceGenerator(
            im_size=self.im_size,
            grid_size=self.grid_size,
            device=self.device,
            dtype=dtype_mapping[self.recopars['dtype']])

        self.kspace_generator.set_traj_generator(generate_golden_angle_radial_sampling_pattern,
                                                 spoke_length=self.im_size[0],
                                                 num_spokes=self.recopars['n_spokes'],
                                                 dtype=self.dtype)

    def set_data_path(self, kdata_path: Path = None, image_path: Path = None, ktraj_path: Path = None,
                      coilmap_path: Path = None):
        self.data['kdata'] = kdata_path
        self.data['ktraj'] = ktraj_path
        self.data['coilmap'] = coilmap_path
        self.data['image'] = image_path

    def prep_data(self):

        # Case 1: full kspace and ktraj is given (full_kspace, mask)

        if (self.data['kdata'] is not None) and (self.data['ktraj'] is not None and (self.data['coilmap']) is not None):
            raise NotImplementedError("Not implemented yet")
            # ToDo : not tested yet
            # self.kdata = load_tensor(self.data['kdata']).to(self.device).to(self.dtype_c).unsqueeze(0)
            # self.target = self.kdata

            # self.coilmap = load_tensor(self.data['coilmap']).to(self.device).to(self.dtype_c).unsqueeze(0)
            # self.ktraj = load_tensor(self.data['ktraj']).to(self.device).to(self.dtype_c).unsqueeze(0)

        # Case 2: image is given, kspace and ktraj are generated
        elif self.data['image'] is not None:
            self.kdata = self.kspace_generator.forward(self.data['image'], self.recopars['n_coilmaps']).to(self.dtype_c)
            self.ktraj = self.kspace_generator.get_trajectory().to(self.dtype)
            self.coilmap = self.kspace_generator.get_coilmaps().to(self.dtype_c)
            self.gt = self.kspace_generator.get_image().to(self.dtype_c)
            self.nuff_op = self.kspace_generator.get_nufft_op()
            self.nuff_op_adj = self.kspace_generator.get_nufft_op_adj()

        self.target = toReal(self.kdata, dim=1).to(self.dtype)  # real valued with batch dimension
        self.x = torch.rand((self.n_channels, self.im_size[0], self.im_size[1])) \
            .to(device=self.device).unsqueeze(0).to(self.dtype)  # randomly generated input
        self.dl = DipDataset(self.x, self.target)

    def prep_model(self):
        self.model = (NCScampiModel.DipUnet(self.n_channels, self.n_channels, produce=False,
                                            imagespace=True, data_consistency=True,
                                            skip_connections=self.recopars['skip_connections'])
                      .to(self.device).to(self.dtype))
        nufft_forward = partial(radial_forward, coilmaps=self.coilmap, ktraj=self.ktraj, nufft_ob=self.nuff_op)
        nufft_backward = partial(radial_backward, im_size=self.im_size, coilmaps=self.coilmap, ktraj=self.ktraj,
                                 nufft_ob_adj=self.nuff_op_adj)
        loss_func = NCScampiLoss(nufft_forward=nufft_forward, nufft_backward=nufft_backward, reduction='mean',
                                 device=self.device)
        optimizer = Adam(params=self.model.parameters(), lr=self.recopars['learningrate'],
                         amsgrad='true'.lower() == self.recopars['amsgrad'].lower())
        trainloader = DataLoader(self.dl)
        self.trainer = Trainer(trainloader, self.model, optimizer, loss_func)

    def forward(self, preview: int = 0):
        self.trainer.train(self.recopars['n_epochs'], name=self.recopars['name'], preview=preview)
        return self.inference()

    def inference(self):
        with torch.no_grad():
            self.model.produce = True
            z = self.model.forward(self.x)
            res = toComplex(z, dim=1).squeeze(0)
        return res

    def get_gt(self):
        if self.gt is None:
            raise ValueError("Ground truth is not available !")
        else:
            return self.gt

    def normalize_kdata(self):
        if self.kdata is None:
            print("No kdata available yet!")
        else:
            self.kdata = self.kdata / torch.max(torch.abs(self.kdata))
