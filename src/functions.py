Wfrom typing import Optional, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn


# Functions for working with image coordinates and labels

def to_onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    Generates one-hot encodings of labels.
    
    Args:
        idx: 
            Tensor of labels
        n: 
            size of vectors (number of labels)
    
    Returns:
        Tensor of one-hot n-dimensional vector encodings of
        the labels in idx.
    """
    if torch.max(idx).item() >= n:
        raise AssertionError(
            "Labelling must start from 0 and "
            "maximum label value must be less than total number of classes")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n, device=device)
    return onehot.scatter_(1, idx.to(device), 1)


def grid2xy(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Helper for imcoordgrid().
    """
    X = torch.cat((X1[None], X2[None]), 0)
    d0, d1 = X.shape[0], X.shape[1] * X.shape[2]
    X = X.reshape(d0, d1).T
    return X


def imcoordgrid(im_dim: Tuple) -> torch.Tensor:
    """
    Generates a list of coordinates for each point on a grid spanning 
    -1 to 1 in both directions.
    
    Args:
        in_dim: Tuple (x,y) of dimensions of the grid
    
    Returns:
        Tensor of shape (x*y, 2) of each coordinate in the grid.
    
    Example:
        >>> imcoordgrid((4,4))
        tensor([[-1.0000,  1.0000],
        [-1.0000,  0.3333],
        [-1.0000, -0.3333],
        [-1.0000, -1.0000],
        [-0.3333,  1.0000],
        ...
        [ 1.0000, -1.0000]])
    """
    xx = torch.linspace(-1, 1, im_dim[0])
    yy = torch.linspace(1, -1, im_dim[1])
    x0, x1 = torch.meshgrid(xx, yy)
    return grid2xy(x0, x1)


def transform_coordinates(
    coord: torch.Tensor,
    phi: Union[torch.Tensor, float] = torch.zeros(1),
    coord_dx: Union[torch.Tensor, float] = 0,
    ) -> torch.Tensor:
    """
    Transforms coordinates by rotation and translation.
    
    Args:
        coord: 
            Tensor of coordinates
        phi: 
            Angle
        coord_dx: 
            Shift
    
    Returns:
        Tensor of translated coordinates
    """
    if torch.sum(phi) == 0:
        phi = coord.new_zeros(coord.shape[0])
    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)
    return coord + coord_dx
