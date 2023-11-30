# Alpha Shape Sampler Paper Results
Code to reproduce analyses and results provided in the alpha shape sampler paper beyond what is given in the R package.

## Directory Contents
The list of directories and their corresponding purpose in this repository include:
* `neutrophilML` has all machine learning experiments for replication of analyses with the neutrophil data, including calculating shape characteristics and performing dimensionality reduction in python. In particular, we show how to implement the manifold regularized autoencoder (MRAE).
* `teeth_demo` contains a subsampled dataset of the primate mandibular molars and a notebook demonstrating how generate a new tooth.
* `mask_to_complex` contains a notebook for converting a binary mask to a simplicial complex and vice versa in R.

## Relevant Citations

D. Bhaskar, D. Lee, H. Knútsdóttir, C. Tan, M. Zhang, P. Dean, C. Roskelley, and L. Keshet. A methodology for morphological feature extraction and unsupervised cell classification. _bioRxiv_. 623793.

E.T. Winn-Nuñez, H. Witt, D. Bhaskar, R. Huang, I.Y. Wong, J. Reichner, and L. Crawford. A probabilistic method for sampling alpha-shapes.
