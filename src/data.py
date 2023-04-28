from typing import Optional, Tuple, Type, Union
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import ToTensor


# Functions for data loading and preprocessing

def get_mnist_data(
    dataset, 
    digits: Tuple[int] = [0,1,2,3,4,5,6,7,8,9], 
    n_samples: int = None, 
    rotation_range: Tuple[int] = None, 
    translation_range: Tuple[int] = None,
    coord: int = None
    ) -> Tuple[torch.Tensor]:
    
    """
    Formats and transforms a dataset in Torch formating. 
    
    Args:
        dataset: 
            Dataset of paired images and labels
        digits: 
            Set of labels to filter from. Defaults to range 0-9.
        n_samples: 
            Maximum amount of samples to use from each label. None if 
            no maximum (default).
        rotation_range: 
            Each image is rotated by a random integer degree chosen from 
            the range. Defaults to None (no rotation).
        translation_range: 
            Each image is translated in by a pair of random integers chosen 
            from the range. Defaults to None (no rotation).
        coord: 
            if specified, overrides rotation_range and translation_range
            with default values, as defined by coord_default.
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
    
    Returns:
        Returns a tuple of tensors representing the transformed data, 
        labels, angles, and shifts.
    """
    
    data, labels, angles, shifts = [], [], [], []
    count = torch.zeros(10)
    
    if coord is not None:
        rotation_range, translation_range = coord_default(coord)
    
    for i, (im, lbl) in enumerate(dataset):
        if lbl in digits:
            
            if n_samples is not None and count[lbl] >= n_samples:
                continue
                
            theta = torch.randint(*rotation_range, (1,)) if rotation_range else torch.zeros((1,))
            dx, dy = np.random.randint(*translation_range, 2) if translation_range else (0,0)

            im = im.rotate(theta, translate = (dx, dy), resample=Image.BICUBIC)

            data.append(ToTensor()(im))
            labels.append(lbl)
            angles.append(torch.deg2rad(theta))
            shifts.append((dx, dy))

            count[lbl] = count[lbl]+1
                
    return torch.cat(data), torch.tensor(labels), torch.tensor(angles), torch.tensor(shifts)

def coord_default(coord: int = None) -> Tuple[Tuple[int]]:
    """
    Specifies default rotation + translation ranges
    to be used in get_mnist_data()
    
    Args:
        coord: 
            integer specifying transformation nature
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
            
    Returns:
        Tuple rotation_range, translation_range.
    """
    if coord not in [0, 1, 2, 3]:
        raise ValueError("'coord' argument must be 0, 1, 2 or 3")
    
    rotation_range = [-60, 61] if coord in [1, 3] else None
    translation_range = [-5, 6] if coord in [2, 3] else None
    
    return rotation_range, translation_range

def init_dataloader(
    *args: torch.Tensor, 
    **kwargs: int
    ) -> Type[torch.utils.data.DataLoader]:
    """
    Creates a Torch dataloader.
    
    Args:
        *args: 
            To be passed into TensorDataset().
            -Torch Tensor data
        batch_size: 
            Keyworded, defaults to 100.
    
    Returns: 
        Torch DataLoader object.
    
    """
    batch_size = kwargs.get("batch_size", 100)
    tensor_set = torch.utils.data.dataset.TensorDataset(*args)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=tensor_set, batch_size=batch_size, shuffle=True)
    
    return data_loader
