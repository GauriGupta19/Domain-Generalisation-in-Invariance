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

SAVE_DIR = "pad_models"

def train_vae(
    trainset,
    name: str,
    data_coord: int,
    model_coord: int,
    labels = [0,1,2,3,4,5,6,7,8,9], 
    batch_size=100, 
    epochs=100,
    **kwargs
    ):
    """
    Trains a VAE.
    
    Args:
        trainset: 
            Raw dataset to train on
        data_coord: 
            Specifies the transformation we apply to the trainset.
            - 0: no transformation
            - 1: rotation only
            - 2: translation only
            - 3: rotation + translation
        model_coord: 
            Specifies the type of invariant VAE.
            - 0: Vanilla VAE
            - 1: Rotation Invariant
            - 2: Translation Invariant
            - 3: Rotation + Translation Invariant
        name: 
            String, saves to path "{SAVE_DIR}/{name}.pkl"
        labels: 
            Labels of dataset to train on. Defaults to 0-9.
        batch_size: 
            Size of batches to train on (larger = more GPU intensive).
        epochs: 
            Number of epochs to train.
        **kwargs: 
            Passed to rVAE() initialization, along with model_coord.
            - in_dim
            - latent_dim
            - num_classes
            - hidden_dim_e
            - hidden_dim_d
            - num_layers_e
            - num_laters_d
            - activation
            - softplus_sd
            - sigmoid_out
            - seed
    """
    
    if name is not None:
        path = f"{SAVE_DIR}/{name}.pkl"
    elif path is None:
        raise NameError("name or path must be specified!")
    
    train_data, train_labels, train_angles, train_shifts = get_mnist_data(
        trainset, 
        digits = labels, 
        coord = data_coord)    
    
    train_loader = init_dataloader(train_data, batch_size=batch_size)
    
    vae = rVAE(coord=model_coord, **kwargs)
    
    trainer = SVItrainer(vae)

    for e in tqdm(range(epochs)):
        trainer.step(train_loader, scale_factor=3)       
    trainer.print_statistics()  
    
    trainer.save_model(vae, path)
    
    
def load_vae(name: str, coord: int, **kwargs):
    """
    Loads and returns a VAE model.
    
    Args:
        name: 
            String name of model located at "saved_models/{name}.pkl"
        coord:
            Specifies the type of VAE to expect.
            - 0: Vanilla VAE
            - 1: Rotation Invariant
            - 2: Translation Invariant
            - 3: Rotation + Translation Invariant
        **kwargs:
            Passed to rVAE() initialization, along with coord. Must match
            the hyperparameters of the saved model.
            - in_dim
            - latent_dim
            - num_classes
            - hidden_dim_e
            - hidden_dim_d
            - num_layers_e
            - num_laters_d
            - activation
            - softplus_sd
            - sigmoid_out
            - seed
    """
    vae = rVAE(coord = coord, **kwargs)
    trainer = SVItrainer(vae)
    trainer.load_model(vae, f'{SAVE_DIR}/{name}.pkl')
    return vae