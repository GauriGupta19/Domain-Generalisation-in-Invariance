# Domain-Generalisation-in-Invariance
This is the GitHub repository originally for the paper submitted at [ICLR 2023](https://iclr.cc/) workshop [PML4DC 2023](https://pml4dc.github.io/iclr2023/). The link to the paper can be found here: [Domain Generalization In Robust Invariant Representation](https://arxiv.org/abs/2304.03431) [[slides](https://drive.google.com/file/d/1HHr7lCHJwNCrxb3oGIwMP0V-8r5pgALj/view)]

The description of different files in the repository can be found below:

* ```ood_new.ipynb``` - The formatted most recent results, generalizing the results of the original paper to more transformations and applications.

* ```src/``` - Core Python code

    * ```data.py``` - Functions for loading, preprocessing, and transforming data.
    * ```functions.py``` - Functions for working with image coordinates and labels.
    * ```models.py``` - Defines functions and the encoder/decoder modules for the VAE frameworks.
    * ```rvae.py``` - The ```rVAE``` class of models used.
    * ```trainer.py``` - The ```SVItrainer``` class used to train the models
    * ```steps.py``` - Convenience functions for training, saving, and loading VAE models.
    
    * ```similarity.py``` - Convenience for performing tests on models.

* ```ood.ipynb``` - The original paper's results which contains the comparisons between the MNIST, FashionMNIST and LFWPeople datasets.

* ```experiments.ipynb``` - The experiments.ipynb file contains the original round of experiments performed, originally from the [pyroVAE_MNIST.ipynb](https://colab.research.google.com/github/ziatdinovmax/notebooks_for_medium/blob/main/pyroVAE_MNIST_medium.ipynb) Colab file.




> The skeleton code for the models has been borrowed from [Explicitly disentangling image content from translation and rotation with spatial-VAE](https://proceedings.neurips.cc/paper/2019/hash/5a38a1eb24d99699159da10e71c45577-Abstract.html) and [How we learnt to love the rotationally invariant variational autoencoders (rVAE)](https://towardsdatascience.com/how-we-learnt-to-love-the-rotationally-invariant-variational-autoencoders-rvae-and-almost-562aa164c59f).

