# Convolutional, Residual, and Fully-connected Networks Provably Contain Lottery Tickets for Most Activation Functions

## Introduction
This Github repository implements the experiments of the IMCL 2022 paper [Convolutional and Residual Networks Provably Contain Lottery Ticket](https://proceedings.mlr.press/v162/burkholz22a.html) and the paper [Most Activation Functions Can Win the Lottery Without Excessive Depth](https://arxiv.org/abs/2205.02321). The main objective is to identify a subnetwork of a randomly initialized neural network, i.e., a strong lottery ticket, that approximates a given target network.
The exemplary target networks were obtained with the help of the Github repositories [Synaptic-Flow](https://github.com/ganguli-lab/Synaptic-Flow) and [open_lth](https://github.com/facebookresearch/open_lth). Correspondingly, we used their definition of the respective neural network architectures.

## License
LT-existence is licensed under the MIT license, as found in the LICENSE file.

## Citation
If you find this repository helpful, please consider citing the two papers it is based on:
```
@InProceedings{pmlr-v162-burkholz22a,
  title = 	 {Convolutional and Residual Networks Provably Contain Lottery Tickets},
  author =       {Burkholz, Rebekka},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {2414--2433},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR}
}
```
and 
```
@misc{act-LT-Burkholz,
  doi = {10.48550/ARXIV.2205.02321},
  author = {Burkholz, Rebekka},
  title = {Most Activation Functions Can Win the Lottery Without Excessive Depth},
  publisher = {arXiv},
  year = {2022}
  }
```
Happy pruning!
