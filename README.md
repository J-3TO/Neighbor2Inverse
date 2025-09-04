# Neighbor2Inverse
Official code implementation of Neighbor2Inverse

Overview of the project: 
https://j-3to.github.io/Neighbor2Inverse/

## Data
We are currently figuring out the best way to share the data. A download link will be shared here.

## Preprocessing, thickness retrieval, and reconstruction
```\Reconstruction``` contains all scripts related to preprocessing and reconstructing the data. Reconstruction uses [torch-radon](https://github.com/matteo-ronchetti/torch-radon). 
The original repo only works with torch<1.8. This forked version works with newer torch versions: [https://github.com/J-3TO/torch-radon.git](https://github.com/J-3TO/torch-radon.git).


### Pipeline: 
```0_preprocessing.py``` : Flat-field, dark-current correction and stitching, processed data is saved as npy files.\
```1_RingArtifactRemoval.py``` : Applies sorting-based sinogram correction proposed by [Vo et al.](https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-22-28396&id=399265)\
```2_PhaseRetrieval.py``` : Applies thickness retrieval proposed by [Paganin et al.](https://onlinelibrary.wiley.com/doi/10.1046/j.1365-2818.2002.01010.x)\
```3_reconstruction.py``` : FBP with padding to account for the ROI measurements.

It is also possible to just run scripts ```0_preprocessing.py``` and ```3_reconstruction.py``` and setting ```phase_retrieval``` and ```ring_removal``` to ```True``` in ```3_reconstruction.py```.
## Neighbor2Inverse

Work in progress, code will be uploaded soon.
