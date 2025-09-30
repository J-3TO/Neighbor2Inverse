# Neighbor2Inverse
[![Interactive Demo](https://img.shields.io/badge/Interactive%20Demo-0099FF
)](https://j-3to.github.io/Neighbor2Inverse/)

Official code implementation of Neighbor2Inverse

## Data
We are currently figuring out the best way to share the data. A download link will be shared here.

## Preprocessing, thickness retrieval, and reconstruction
```\Reconstruction``` contains all scripts related to preprocessing and reconstructing the data. Reconstruction uses [torch-radon](https://github.com/matteo-ronchetti/torch-radon). 
The original repo only works with torch<1.8. This forked version works with newer torch versions: [https://github.com/J-3TO/torch-radon.git](https://github.com/J-3TO/torch-radon.git).


### Pipeline: 
```0_preprocessing.py``` : Flat-field, dark-current correction and stitching, processed data is saved as npy files.\
```1_RingArtifactRemoval.py``` : Applies sorting-based sinogram correction proposed by [Vo et al.](https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-22-28396&id=399265). This filtering was only used for the testing data, not for the training split\
```2_PhaseRetrieval.py``` : Applies thickness retrieval proposed by [Paganin et al.](https://onlinelibrary.wiley.com/doi/10.1046/j.1365-2818.2002.01010.x)\
```3_reconstruction.py``` : FBP with padding to account for the ROI measurements.

It is also possible to just run scripts ```0_preprocessing.py``` and ```3_reconstruction.py``` and setting ```phase_retrieval``` and ```ring_removal``` to ```True``` in ```3_reconstruction.py```.

## Neighbor2Inverse
First run ```0_calculateStats``` to get mean&std for each stack for normalization and save it in csv file.\
To train run ```1_trainNeighbor2Inverse.py``` with the respective trainingparams.yml file:\

```python 1_reconstruction.py --trainparams trainparamsNeighbor2InverseProjSub.yml``` : Trains Neighbor2Inverse with proj subsampling and regularization term\

```python 1_reconstruction.py --trainparams trainparamsNeighbor2InverseSinoSub.yml``` : Trains Neighbor2Inverse with sino subsampling without regularization term\

```python 1_reconstruction.py --trainparams trainparamsNeighbor2InverseDataVidelityOrigSino.yml``` : Trains Neighbor2Inverse with the origSino data fidelity term\

```python 1_reconstruction.py --trainparams trainparamsNeighbor2InverseDataVidelityVirtSino.yml``` : Trains Neighbor2Inverse with the virtSino data fidelity term\

```python 1_reconstruction.py --trainparams trainparamsSparse.yml``` : Trains Neighbor2Inverse without regularization and with sparse-sampling of 900 projections





