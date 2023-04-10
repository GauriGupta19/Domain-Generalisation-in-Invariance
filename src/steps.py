from tqdm import tqdm
from typing import Optional, Tuple, Type, Union
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore", module="torchvision.datasets")

from src.models import *
from src.functions import *
from src.data import *
from src.rvae import *
from src.trainer import *

from tqdm import tqdm
def train_vae(
    trainset,  
    coord,
    model_coord,
    path,
    labels = [0,1,2,3,4,5,6,7,8,9], 
    latent_dim = 10,
    in_dim=(28,28), 
    hidden_dim_e=128, 
    hidden_dim_d=128,
    batch_size=100, 
    epochs=100
    ):
    
    if coord not in [0, 1, 2, 3]:
            raise ValueError("'coord' argument must be 0, 1, 2 or 3")
    
    if coord == 1 or coord == 3:
        rotation_range = [-60, 61]
    else:
        rotation_range = [0,1]
    if coord == 2 or coord == 3:
        translation_range = [-10, 11]
    else:
        translation_range = [0,1]
    
    train_data, train_labels, train_angles, train_translations = get_mnist_data(
        trainset, 
        digits = labels, 
        rotation_range=rotation_range, 
        translation_range=translation_range)
    
    train_loader = init_dataloader(train_data, batch_size=batch_size)
    
    # # Initialize probabilistic VAE model ->
    # # (coord=0: vanilla VAE
    # #  coord=1: rotations only
    # #  coord=2: translations only
    # #  coord=3: rotations+translations)

    vae = rVAE(in_dim, latent_dim=latent_dim, coord=model_coord, seed=0, hidden_dim_e=hidden_dim_e, hidden_dim_d=hidden_dim_d)

    # # # Initialize SVI trainer
    trainer = SVItrainer(vae)
    # Train for n epochs:ccc

    for e in tqdm(range(epochs)):
        # Scale factor balances the qualitiy of reconstruction with the quality of disentanglement
        # It is optional, and the rvae will also work without it
        trainer.step(train_loader, scale_factor=3)
        trainer.print_statistics()
    trainer.save_model(vae, path)
    
#     trainer.load_model(rvae, path)