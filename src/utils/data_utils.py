__author__ = "Volker Herold"
__year__ = "2023"
__version__ = "0.0.1"

import os
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import random
from typing import Union, List, Tuple
import scipy.io as sio
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset


def load_tensor(filename: Union[str, Path] = None):
    """
    Load a tensor from a file.

    If the file is a numpy file (.npy), load it as a numpy array and convert to a torch tensor.
    If the file is a torch file (.pt), load it as a torch tensor.

    Args:
    filename (str): Filename without extension.

    Returns:
    tensor: The loaded tensor.
    """
    # Convert filename to string
    filename = str(filename)

    # Check if numpy file exists
    if os.path.isfile(filename + '.npy'):
        array = np.load(filename + '.npy')
        tensor = torch.from_numpy(array)

    # Check if torch file exists
    elif os.path.isfile(filename + '.pt'):
        tensor = torch.load(filename + '.pt')

    else:
        raise FileNotFoundError(f"No file found with name {filename}.npy or {filename}.pt")

    return tensor


def mda_slice(array, dims, fixed_index=0):
    """
    Take a slice from a multidimensional numpy array or PyTorch tensor along the specified dimensions.

    All other dimensions are fixed at a specified index.

    Args:
    array (np.ndarray or torch.Tensor): The input array or tensor.
    dims (tuple of int): The dimensions to include in the output.
    fixed_index (int): The index at which to fix all other dimensions.

    Returns:
    slice_result: The extracted slice.
    """

    # Create a list of slice(None) for all dimensions
    slicers = [slice(None)] * array.ndim

    # translate negative dimensions
    dim_list = []

    for i in dims:
        if i < 0:
            d = array.ndim + i
        else:
            d = i
        dim_list.append(d)
    dims = tuple(dim_list)

    # For dimensions not in dims, fix the slicer at the fixed_index
    for i in range(array.ndim):
        if i not in dims:
            slicers[i] = slice(fixed_index, fixed_index + 1, 1)

    # Convert slicers list to a tuple and use it to slice the array
    slice_result = array[tuple(slicers)].squeeze()

    return slice_result


def load_resize_image(image_path, output_size):
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # convert to grayscale
        transforms.Resize(output_size),  # resize
        transforms.ToTensor(),  # convert to tensor
    ])

    # Load image
    # image = Image.open(image_path).transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
    image = Image.open(image_path)

    # Apply the transformations
    tensor = transform(image)

    # Convert the PyTorch tensor to a complex-valued tensor
    tensor = torch.view_as_complex(
        torch.stack((tensor.to(torch.float32), tensor.to(torch.float32)), dim=-1))

    # Normalize the tensor
    tensor = tensor / torch.max(torch.abs(tensor))

    return tensor


def rndraw(filelist: List[str], k: int = 1):
    """Draw k filenames randomly from filelist and load torch tensors."""
    draw = []
    for f in random.choices(filelist, k=k):
        draw.append(torch.load(f))
    return torch.stack(draw, dim=0)


def toComplex(input: torch.Tensor, dim: int) -> torch.complex64:
    """
    Convert a real-valued PyTorch tensor to a complex-valued tensor.

    Args:
        input (torch.Tensor): The input real-valued tensor.
        dim (int): The dimension along which to convert the tensor to complex.

    Returns:
        torch.Tensor: The complex-valued tensor.
    """
    n_dims = len(input.shape)

    if dim >= n_dims or dim < 0:
        raise ValueError(f"Invalid dimension {dim}. Tensor has {len(input.shape)} dimensions.")

    if input.shape[dim] > 2 and input.shape[dim] % 2 == 0:
        new_shape = list(input.shape)
        new_shape[dim:dim + 1] = [2, int(input.shape[dim] / 2)]
        input = input.reshape(tuple(new_shape))
    elif input.shape[dim] == 2:
        pass
    else:
        raise ValueError(f"Invalid dimension {dim}.Tensor does not have an even size or a size of 2 along this "
                         f"dimension.")

        # Create a slice object that slices all dimensions except the specified one
    slice_indices_real = [slice(None)] * len(input.shape)
    slice_indices_real[dim] = slice(0, 1, 1)
    slice_indices_imag = [slice(None)] * len(input.shape)
    slice_indices_imag[dim] = slice(1, 2, 1)

    # Use the slice object to slice the tensor along the specified dimension
    sliced_tensor_real = (input[slice_indices_real]).squeeze(dim)
    sliced_tensor_imag = (input[slice_indices_imag]).squeeze(dim)

    res = torch.view_as_complex(
        torch.stack((sliced_tensor_real.to(torch.float32), sliced_tensor_imag.to(torch.float32)), dim=-1))

    if res.dim() < n_dims:  # avoid dim collaps
        res = res.unsqueeze(dim)

    return res


def toReal(input: torch.Tensor, dim: int):
    return torch.cat((input.real, input.imag), dim=dim)


def matlab_to_torch(filename, varname):
    # Load the MATLAB file
    mat_data = sio.loadmat(filename)

    # Identify the variable you want to convert to a PyTorch tensor
    numpy_data = mat_data[varname]

    # Convert the NumPy array to a PyTorch tensor
    torch_data = torch.from_numpy(numpy_data)

    return torch_data


def fdiff(input: torch.Tensor, axes: Union[List, Tuple]) -> torch.Tensor:
    """Linear operator that computes finite difference gradient.

        Args:
            input (torch.Tensor): Input tensor.
            axes (tuple or list): Axes to circularly shift. All axes are used if
                None.

      """
    output = torch.empty(len(axes), *tuple(input.shape), dtype=input.dtype)
    for idx, i in enumerate(axes):
        output[idx, :] = input - torch.roll(input, 1, i)

    return output


class DipDataset(Dataset):
    """A minimal working torch Dataset.

    This dataset can be used alternatively to DIPTransform, when input and target are already loaded as tensors.
    Then the DipDataset is just constructed by calling the __init__() function.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        """A minimal working torch Dataset.

        :param x: Input.
        :param y: Target.
        """
        super(DipDataset, self).__init__()
        # store the raw tensors
        self._x = x
        self._y = y

    def __len__(self):
        # a DataSet must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y


class Trainer:
    def __init__(self, trainloader, model, optimizer, loss_func):
        """Set up a trainer, that wraps everything to train a model.

        Trainer.train() trains the model.

        :param trainloader: Torch dataloader for training.
        :param model: Torch model.
        :param optimizer: Torch optimizer function.
        :param loss_func: Torch loss function.
        """
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.running_loss = 0

    def train(self, epochs, name: str = "", preview: int = 0):

        pbar = tqdm(total=epochs, position=0, leave=False)
        pbar.set_description(f"{name} Training: ")
        outputs = None
        for epoch in range(epochs):
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                sc = self.model.skip_connections
                pbar.set_postfix({'loss': f'{loss:.4f}', 'Skip': f'{sc:.3f}'}, refresh=True)
                pbar.update()

                # Turn on interactive mode
            if preview > 0 and epoch % preview == 0:
                im = torch.abs(mda_slice(toComplex(outputs, dim=1), dims=[-2, -1]))
                im = im.cpu().detach().numpy()
                plt.imshow(im, cmap='gray')
                plt.show()
                # print(f'Skip connections: {sc}')

        plt.ioff()  # Turn off interactive mode
        pbar.close()
        return self.model


def sobel_operator(input):
    input_real = input.real  # Extract the real part of the complex tensor
    input_imag = input.imag  # Extract the imaginary part of the complex tensor

    kernel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)
    kernel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).unsqueeze(
        0).unsqueeze(0)

    kernel_x = kernel_x.to(input.device)
    kernel_y = kernel_y.to(input.device)

    sobel_x_real = F.conv2d(input_real, kernel_x, padding=1)
    sobel_y_real = F.conv2d(input_real, kernel_y, padding=1)
    sobel_x_imag = F.conv2d(input_imag, kernel_x, padding=1)
    sobel_y_imag = F.conv2d(input_imag, kernel_y, padding=1)

    sobel_output_real = torch.sqrt(sobel_x_real ** 2 + sobel_y_real ** 2)
    sobel_output_imag = torch.sqrt(sobel_x_imag ** 2 + sobel_y_imag ** 2)

    return sobel_output_real + sobel_output_imag
