#@title Load utility functions for data loading and preprocessing
from typing import Optional, Tuple, Type, Union
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

def get_rotated_mnist(dataset, rotation_range: Tuple[int]) -> Tuple[torch.Tensor]:
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    imstack_data_r = torch.zeros_like(dataset.data, dtype=torch.float32)
    labels, angles = [], []
    for i, (im, lbl) in enumerate(dataset):
        theta = torch.randint(*rotation_range, (1,)).float()
        im = im.rotate(theta.item(), resample=Image.BICUBIC)
        imstack_data_r[i] = ToTensor()(im)
        labels.append(lbl)
        angles.append(torch.deg2rad(theta))
    imstack_data_r /= imstack_data_r.max()
    return imstack_data_r, torch.tensor(labels), torch.tensor(angles)

def get_rotated_mnist_sample(dataset, n_samples, rotation_range: Tuple[int]) -> Tuple[torch.Tensor]:
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    data, labels, angles = [], [], []
    count = torch.zeros(10)
    for i, (im, lbl) in enumerate(dataset):
        if count[lbl]<n_samples:
            theta = torch.randint(*rotation_range, (1,)).float()
            im = im.rotate(theta.item(), resample=Image.BICUBIC)
            data.append(ToTensor()(im))
            labels.append(lbl)
            angles.append(torch.deg2rad(theta))
            count[lbl] = count[lbl]+1
    return torch.cat(data), torch.tensor(labels), torch.tensor(angles)

def get_translated_mnist(dataset, translation_range: Tuple[int]) -> Tuple[torch.Tensor]:
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    imstack_data_r = torch.zeros_like(dataset.data, dtype=torch.float32)
    labels, translations = [], []
    for i, (im, lbl) in enumerate(dataset):
        theta = 0
        translate_1 = np.random.randint(*translation_range)
        translate_2 = np.random.randint(*translation_range)
        im = im.rotate(theta, translate = (translate_1, translate_2), resample=Image.BICUBIC)
        imstack_data_r[i] = ToTensor()(im)
        labels.append(lbl)
        translations.append((translate_1, translate_2))
    imstack_data_r /= imstack_data_r.max()
    return imstack_data_r, torch.tensor(labels), torch.tensor(translations)

def get_rotated_translated_mnist(dataset, rotation_range: Tuple[int], translation_range: Tuple[int]) -> Tuple[torch.Tensor]:
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    imstack_data_r = torch.zeros_like(dataset.data, dtype=torch.float32)
    labels, angles, translations = [], [], []
    for i, (im, lbl) in enumerate(dataset):
        theta = torch.randint(*rotation_range, (1,)).float()
        translate_1 = np.random.randint(*translation_range)
        translate_2 = np.random.randint(*translation_range)
        im = im.rotate(theta.item(), translate = (translate_1, translate_2), resample=Image.BICUBIC)
        imstack_data_r[i] = ToTensor()(im)
        labels.append(lbl)
        angles.append(torch.deg2rad(theta))
        translations.append((translate_1, translate_2))
    imstack_data_r /= imstack_data_r.max()
    return imstack_data_r, torch.tensor(labels), torch.tensor(angles), torch.tensor(translations)


def init_dataloader(*args: torch.Tensor, **kwargs: int
                    ) -> Type[torch.utils.data.DataLoader]:

    batch_size = kwargs.get("batch_size", 100)
    tensor_set = torch.utils.data.dataset.TensorDataset(*args)
    data_loader = torch.utils.data.DataLoader(
        dataset=tensor_set, batch_size=batch_size, shuffle=True)
    return data_loader
