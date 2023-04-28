from typing import Optional, Tuple, Type, Union
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import pyro.optim as optim
import torch
import torch.nn as nn
from scipy.stats import norm
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", module="torchvision.datasets")

from src.models import *
from src.functions import *

class rVAE(nn.Module):
    """
    Variational autoencoder with rotational and/or translational invariance
    """
    def __init__(self,
                 in_dim: Tuple[int] = (28, 28),
                 latent_dim: int = 10,
                 coord: int = 3,
                 num_classes: int = 0,
                 hidden_dim_e: int = 128,
                 hidden_dim_d: int = 128,
                 num_layers_e: int = 2,
                 num_layers_d: int = 2,
                 activation: str = "tanh",
                 softplus_sd: bool = True,
                 sigmoid_out: bool = True,
                 seed: int = None,
                 **kwargs
                 ) -> None:
        """
        Initializes rVAE's modules and parameters.
        
        Args:
            in_dim: 
                Shape of input data. Defaults to (28, 28)
            latent_dim:
                default 10
            coord:
                Specifies the type of invariant VAE.
                - 0: Vanilla VAE
                - 1: Rotation Invariant
                - 2: Translation Invariant
                - 3: Rotation + Translation Invariant
            num_classes:
                Only for training class-conditioned VAE.
                Specifies number of classes to label.
            hidden_dim_e: default 128
            hidden_dim_d: default 128
            activation: default tanh
            softplus_sd: default True
            sigmoid_out: default True
            seed: default None
                
        """
        super().__init__()
        pyro.clear_param_store()
        if seed is not None:
            set_deterministic_mode(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.encoder_net = fcEncoderNet(
            in_dim, latent_dim+coord, hidden_dim_e,
            num_layers_e, activation, softplus_sd)
        
        if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
        dnet = rDecoderNet if coord in [1, 2, 3] else fcDecoderNet
        self.decoder_net = dnet(
            in_dim, latent_dim+num_classes, hidden_dim_d,
            num_layers_d, activation, sigmoid_out)
        
        self.z_dim = latent_dim + coord
        self.coord = coord
        self.num_classes = num_classes
        self.grid = imcoordgrid(in_dim).to(self.device)
        self.dx_prior = torch.tensor(kwargs.get("dx_prior", 0.1)).to(self.device)
        self.to(self.device)

    def model(self,
              x: torch.Tensor,
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the Pyro model p(x|z)p(z).
        
        Args:
            x: 
                Batch input data
            y: 
                Labels (for class conditional VAE)
        """
        # register PyTorch module `decoder_net` with Pyro
        pyro.module("decoder_net", self.decoder_net)
        
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        
        reshape_ = torch.prod(torch.tensor(x.shape[1:])).item()
        with pyro.plate("data", x.shape[0]):
            
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            
            # sample from prior (value will be sampled by guide when computing the ELBO)
            with pyro.poutine.scale(scale=beta):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            # rotationally- and/or translationaly-invariant mode
            if self.coord > 0:  
                
                # Split latent variable into parts for transformation and image content
                phi, dx, z = self.split_latent(z)
                if torch.sum(dx) != 0:
                    dx = (dx * self.dx_prior).unsqueeze(1)
                    
                # transform coordinate grid
                grid = self.grid.expand(x.shape[0], *self.grid.shape)
                x_coord_prime = transform_coordinates(grid, phi, dx)
                
            # Add class label (if any)
            if y is not None:
                y = to_onehot(y, self.num_classes)
                z = torch.cat([z, y], dim=-1)
                
            # decode the latent code z together with the transformed coordinates (if any)
            dec_args = (x_coord_prime, z) if self.coord else (z,)
            loc_img = self.decoder_net(*dec_args)
            
            # score against actual images ("binary cross-entropy loss")
            pyro.sample(
                "obs", dist.Bernoulli(loc_img.view(-1, reshape_), validate_args=False).to_event(1),
                obs=x.view(-1, reshape_))
            
    def guide(self,
              x: torch.Tensor,
              y: Optional[torch.Tensor] = None,
              **kwargs: float) -> torch.Tensor:
        """
        Defines the Pyro guide q(z|x).
        
        Args:
            x: 
                Batch input data
            y: 
                Labels (for class conditional VAE)
        """
        # register PyTorch module `encoder_net` with Pyro
        pyro.module("encoder_net", self.encoder_net)
        
        # KLD scale factor (see e.g. https://openreview.net/pdf?id=Sy2fzU9gl)
        beta = kwargs.get("scale_factor", 1.)
        
        with pyro.plate("data", x.shape[0]):
            
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder_net(x)
            
            # sample the latent code z
            with pyro.poutine.scale(scale=beta):
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def split_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Split latent variable into parts for rotation and/or translation 
        and image content. Depending on self.coord, it truncates and returns
        the first 0-3 values in z (filling in the blanks with zeroes).
        """
        phi, dx = torch.tensor(0), torch.tensor(0)
        # rotation + translation
        if self.coord == 3: 
            phi = z[:, 0]  # encoded angle
            dx = z[:, 1:3]  # translation
            z = z[:, 3:]  # image content
        # translation only
        elif self.coord == 2:
            dx = z[:, :2]
            z = z[:, 2:]
        # rotation only
        elif self.coord == 1: 
            phi = z[:, 0]
            z = z[:, 1:]
        return phi, dx, z
    
    def _encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        in a batch-by-batch fashion.
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                encoded = self.encoder_net(x_i)
            encoded = torch.cat(encoded, -1).cpu()
            return encoded

        x_new = x_new.to(self.device)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(x_new) // num_batches
        z_encoded = []
        for i in range(num_batches):
            x_i = x_new[i*batch_size:(i+1)*batch_size]
            z_encoded_i = inference()
            
            z_encoded.append(z_encoded_i)
        x_i = x_new[(i+1)*batch_size:]
        if len(x_i) > 0:
            z_encoded_i = inference()
            z_encoded.append(z_encoded_i)
        return torch.cat(z_encoded)

    def encode(self, x_new: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Encodes data using a trained inference (encoder) network
        (this is baiscally a wrapper for self._encode).
        """
        if isinstance(x_new, torch.utils.data.DataLoader):
            x_new = train_loader.dataset.tensors[0]
        z = self._encode(x_new)
        z_loc = z[:, :self.z_dim]
        z_scale = z[:, self.z_dim:]
        return z_loc, z_scale
    
    def decode(self, z: torch.Tensor, **kwargs: int) -> torch.Tensor:
        """
        Decodes data using a trained inference (decoder) network
        in a batch-by-batch fashion.
        """
        def inference() -> np.ndarray:
            with torch.no_grad():
                z_i = z_sample_i[:,self.coord:]
                d_args = (self.grid.expand(z_i.shape[0], *self.grid.shape), z_i) if self.coord > 0 else (z_i,)
                decoded = self.decoder_net(*d_args).cpu()
            # decoded = torch.cat(decoded, -1).cpu()
            return decoded

        z_mu, z_sigma = z[0], z[1]
        z_sample = z_mu + torch.randn_like(z_mu)*z_sigma
        z_sample = z_sample.to(self.device)
        num_batches = kwargs.get("num_batches", 10)
        batch_size = len(z_sample) // num_batches
        x_decoded, angles_decoded, translations_decoded = [], [], []
        
        for i in range(num_batches):
            z_sample_i = z_sample[i*batch_size:(i+1)*batch_size]
            x_decoded_i = inference()
            x_decoded.append(x_decoded_i)
        z_sample_i = z_sample[(i+1)*batch_size:]
        if len(z_sample_i) > 0:
            x_decoded_i = inference()
            x_decoded.append(x_decoded_i)
        return torch.cat(x_decoded)


    def manifold2d(self, d: int, **kwargs: Union[str, int]) -> torch.Tensor:
        """
        Plots a learned latent manifold in the image space.
        """
        if self.num_classes > 0:
            cls = torch.tensor(kwargs.get("label", 0))
            cls = to_onehot(cls.unsqueeze(0), self.num_classes)
        grid_x = norm.ppf(torch.linspace(0.95, 0.05, d))
        grid_y = norm.ppf(torch.linspace(0.05, 0.95, d))
        loc_img_all = []
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = torch.tensor([xi, yi]).float().to(self.device).unsqueeze(0)
                if self.num_classes > 0:
                    z_sample = torch.cat([z_sample, cls], dim=-1)
                d_args = (self.grid.unsqueeze(0), z_sample) if self.coord > 0 else (z_sample,)
                loc_img = self.decoder_net(*d_args)
                loc_img_all.append(loc_img.detach().cpu())
        loc_img_all = torch.cat(loc_img_all)

        grid = make_grid(loc_img_all[:, None], nrow=d,
                         padding=kwargs.get("padding", 2),
                         pad_value=kwargs.get("pad_value", 0))
        plt.figure(figsize=(8, 8))
        plt.imshow(grid[0], cmap=kwargs.get("cmap", "gnuplot"),
                   origin=kwargs.get("origin", "upper"),
                   extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("$z_1$", fontsize=18)
        plt.ylabel("$z_2$", fontsize=18)
        plt.show()
