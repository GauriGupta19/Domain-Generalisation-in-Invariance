# Domain-Generalisation-in-Invariance
This is the GitHub repository for the paper submitted at [ICLR 2023](https://iclr.cc/) workshop [PML4DC 2023](https://pml4dc.github.io/iclr2023/). The link to the paper can be found here: [Domain Generalization In Robust Invariant Representation]()

The description of different files in the repository can be found below:
* ```experiments.ipynb``` - The experiments.ipynb file contains the experiments along with the visualizations of the learned latent manifold for transformations like rotation and transformation.
* ```ood.ipynb``` - The results can be generated by running the file ood.ipynb which contains the comparisons between the MNIST, FashionMNIST and LFWPeople datasets.
* ```sample_complexity.ipynb``` - The sample complextiy analysis between Vanilla VAE and Rotationally-invariant VAE (rVAE). Here, rVAE refers to the RotInvVAE model discussed in the paper
> The skeleton code for the experiments has been borrowed from [Explicitly disentangling image content from translation and rotation with spatial-VAE](https://proceedings.neurips.cc/paper/2019/hash/5a38a1eb24d99699159da10e71c45577-Abstract.html)

