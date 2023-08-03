# PP-MPI
Official repository for Deep Equilibrium MPI Reconstruction with Learned Consistency (DEQ-MPI)

A. Güngör, B. Askin, D. A. Soydan, C. B. Top, E. U. Saritas and T. Çukur, "DEQ-MPI: A Deep Equilibrium Reconstruction with Learned Consistency for Magnetic Particle Imaging," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2023.3300704.


# Demo
You can use the following links to download training, validation, test datasets. 

# Dataset

Download and put the datasets folder in the same folder as the code:

https://drive.google.com/drive/folders/1tTRM-vhqONWCkZsW2Gxd5MZi8L-U13N6?usp=sharing


# Pretrained Networks:
Pre-trained network can be found under training/ folder.

# Training

There are three different training codes. Two of them for pre-training of the denoiser networks, one for end-to-end training of the DEQ-MPI network.

Pretraining denoiser network code:

```python train_ppmpi.py --useGPU 0```

```useGPU: Selected GPU
wd: weight decay, default is 0
lr: learning rate
saveModelEpoch: save model every X epoch
valEpoch: Compute validation per X epoch
fixedNsStdFlag: 0: randomly generate noise std for each image, 1: fix noise std.
minNoiseStd: For non-fixed noise, minimum noise std.
maxNoiseStdList: For non-fixed noise: maximum noise std., For fixed noise: noise std. Multiple inputs are separated by comma, each input trains a different network consecutively.
batch_size_train: batch size
epoch_nb: number of epochs
wandbFlag: use Wandb for loss tracking (0 / 1)
wandbName: Experiment name for wandb
reScaleBetween: "x,y" rescale images in the dataset between x and y
dims: 2, number of dimensions of the image

nb_of_featuresList: Number of features of RDN, separate with comma for training of multiple different networks
nb_of_blocks: Number of blocks of RDN
layer_in_each_block: Layer in each block of RDN
growth_rate: growth rate of RDN
```

Pretraining learned consistency network code:

```python train_dcDenoiser.py --useGPU 0```

```useGPU: Selected GPU
wd: weight decay, default is 0
lr: learning rate
saveModelEpoch: save model every X epoch
valEpoch: Compute validation per X epoch
fixedNsStdFlag: 0: randomly generate noise std for each image, 1: fix noise std.
pSNRdataList: input pSNR list, separate with comma for training of multiple different networks.
lrEpoch: Update learning rate every X epoch

batch_size_trainInpList: batch size list, separate with comma for training of multiple different networks.
epoch_nb: number of epochs
wandbFlag: use Wandb for loss tracking (0 / 1)
wandbName: Experiment name for wandb
mtxCode: matrix used for generating the data
reScaleBetween: "x,y" rescale images in the dataset between x and y
reScaleEpsilon: rescale epsilon value in inference by. Higher scaling may help improve performance for high pSNR

useDCNormalization: Used Data Consistency Normalization Type: 0: No normalization: 1 proposed normalization
nb_of_featuresLList: Number of features of learned consistency network, separate with comma for training of multiple different networks
nb_of_blocksLList: Number of blocks of learned consistency network, separate with comma for training of multiple different networks

noisySysMtx: Add noise to system matrix
consistencyDim: Dimensionality of the learned data consistency: 0: conventional consistency, 1: 1D consistency
useLoss: 0: l1-loss, 1: l2-loss
```


DEQ-MPI training code code:

```python train_deqmpi.py --useGPU 2 --epoch_nb 200 --valEpoch 2 --pSNRdataList 15 --reScaleBetween 0.5,1.5 --mtxCode ./inhouseData/expMatinHouse.mat --lrEpoch 150 --consistencyDim 1 --preLoadDir ./training/denoiser/ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn0.5_1.0/epoch200END.pth --preLoadDirDC ./training/dcDenoiser/dcDenoiserPsi_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_18.0_fixNs_1_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_sN_0.02_ls_0/epoch200END.pth```


```useGPU: Selected GPU
wd: weight decay, default is 0
lr: learning rate
saveModelEpoch: save model every X epoch
valEpoch: Compute validation per X epoch
batch_size_train: batch size
epoch_nb: number of epochs
wandbFlag: use Wandb for loss tracking (0 / 1)
wandbName: Experiment name for wandb
optionalString: Optional naming prefix for WanDB and saving model

fixedNsStdFlag: 0: randomly generate noise std for each image, 1: fix noise std.
pSNRdataList: input pSNR list, separate with comma for training of multiple different networks.
mtxCode: matrix used for generating the data
nbOfSingulars: Number of singular values used for least squares initialization

reScaleBetween: "x,y" rescale images in the dataset between x and y
reScaleEpsilon: rescale epsilon value in inference by. Higher scaling may help improve performance for high pSNR


nb_of_featuresList: Number of features of denoiser, separate with comma for training of multiple different networks
nb_of_blocks: Number of blocks of denoiser
layer_in_each_block: Layer in each block of denoiser
growth_rate: growth rate of denoiser

nb_of_steps: Number of steps. ONLY used for unrolled variant

consistencyDimList: Dimensionality list of the learned data consistency: 0: conventional consistency, 1: 1D consistency
nb_of_featuresLList: Number of features of learned consistency network, separate with comma for training of multiple different networks
nb_of_blocksL: Number of blocks of learned consistency network
useDCNormalization: Used Data Consistency Normalization Type: 0: No normalization: 1 proposed normalization

preLoadDir: preload denoiser network path
preLoadDirDC: preload learned consistency network path

```

# Inference for Simulated & Experimental datasets

Simulated:

```python inferenceSimulated.py```

Experimental:

```python inferenceExperimental.py```

Settings should be changed from within the file. A jupyter notebook might be more helpful since it also helps better visualize and manipulate reconstructed images.


**************************************************************************************************************************************
# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```

@ARTICLE{deqmpi,
  author={Güngör, Alper and Askin, Baris and Soydan, Damla Alptekin and Top, Can Barış and Saritas, Emine Ulku and Çukur, Tolga},
  journal={IEEE Transactions on Medical Imaging}, 
  title={DEQ-MPI: A Deep Equilibrium Reconstruction with Learned Consistency for Magnetic Particle Imaging}, 
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}}


@InProceedings{ppmpi,
author="Askin, Baris
and G{\"u}ng{\"o}r, Alper
and Alptekin Soydan, Damla
and Saritas, Emine Ulku
and Top, Can Bar{\i}{\c{s}}
and Cukur, Tolga",
editor="Haq, Nandinee
and Johnson, Patricia
and Maier, Andreas
and Qin, Chen
and W{\"u}rfl, Tobias
and Yoo, Jaejun",
title="PP-MPI: A Deep Plug-and-Play Prior for Magnetic Particle Imaging Reconstruction",
booktitle="Machine Learning for Medical Image Reconstruction",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="105--114",
isbn="978-3-031-17247-2"
}


```
(c) ICON Lab 2023

# Prerequisites

- Python 3.8.10
- CuDNN 8.2.1
- PyTorch 1.10.0

# Acknowledgements

For questions/comments please send an email to: alperg@ee.bilkent.edu.tr