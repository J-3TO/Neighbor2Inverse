import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
import os
    
class ProjDatasetSlice(torch.utils.data.Dataset):
    #Load projection (with option for sparse sampling) and corresponding reconstructed slice data

    def __init__(self, path_proj, path_reco, df_path, df_stats_path, exptime, n_slices, skip=1, n_slicesPR=5, sparseSampling=1):
        
        self.df = pd.read_csv(df_path)
        self.positions = [int(elem) for elem in self.df['position'].values]
        self.total_n_slices = 2149
        
        if exptime=='15ms':
            self.positions = [int(elem) for elem in list(self.positions) if 0 != elem]
        self.identifier_list = [f'pos:{pos}_slIndex:{slice_index}' for pos in self.positions for slice_index in range(0, self.total_n_slices+1)][::skip]

        self.path_proj = path_proj
        self.path_reco = path_reco
        self.exptime = exptime
        self.df_stats = pd.read_csv(df_stats_path)
        self.n_slices = n_slices
        self.n_slicesPR = n_slicesPR
        self.sparseSampling = sparseSampling

    def __len__(self):
        return len(self.identifier_list)

    def __getitem__(self, idx):
        identifier = self.identifier_list[idx % len(self.identifier_list)]
        ids = identifier.split('_')
       
        pos, slice_index = int(ids[0].split(':')[1]), int(ids[1].split(':')[1])
        #print('get proj stack')
        proj_stack, slice_stack = self.load(pos, slice_index)

        return proj_stack, slice_stack, pos, self.exptime

    def load(self, pos, slice_index, exptime=None):
        total_slices = self.total_n_slices

        # Adjust slice index if needed
        if slice_index + self.n_slicesPR - 1 > total_slices:
            slice_index = total_slices - self.n_slicesPR + 1

        # Create slice selection once
        proj_selection = (
            slice(None),
            slice(None),
            slice(slice_index, slice_index + self.n_slicesPR),
            slice(None)
        )

        difference = self.n_slicesPR - self.n_slices
        slice_index_CTStack = slice_index + (difference + 1) // 2  #load neighbouring slices that get subsampled together
        
        slice_selection = (
            slice(slice_index_CTStack, slice_index_CTStack + self.n_slices),
            slice(None),
            slice(None),
            slice(None)
        )

        if exptime is None:
            exptime = self.exptime

        proj_stack = np.load(f'{self.path_proj}/projStitched_{exptime}_pos{pos}.npy', mmap_mode='c')[proj_selection]
        proj_stack = proj_stack.astype('float32')

        # Ensure the last dimension of proj_stack is even
        if proj_stack.shape[-1] % 2 != 0:
            proj_stack = proj_stack[..., :-1]

        # If sparse-sampling, just just retun the thickness retrieved projections and reco the sparse-view reconstruction on the fly
        if self.sparseSampling > 1:
            proj_stack = proj_stack[::self.sparseSampling]
            
            proj_stack_pr = np.load(os.path.dirname(self.path_proj) + f'/projectionsPR/projPR_{exptime}_pos{pos}.npy', mmap_mode='c')[proj_selection][::self.sparseSampling]
            proj_stack_pr = proj_stack_pr.astype('float32')

            if proj_stack_pr.shape[-1] % 2 != 0:
                proj_stack_pr = proj_stack_pr[..., :-1]

            return proj_stack[:, 0], proj_stack_pr[:, 0]

        else:

            slice_stack = np.load(f'{self.path_reco}/reco_{exptime}_pos{pos}.npy', mmap_mode='c')[slice_selection]
            slice_stack = slice_stack.astype('float32')

            # Ensure the last two dimensions of slice_stack are even
            if slice_stack.shape[-2] % 2 != 0:
                slice_stack = slice_stack[..., :-1, :]
            if slice_stack.shape[-1] % 2 != 0:
                slice_stack = slice_stack[..., :, :-1]

            # Return the processed stacks
            return proj_stack[:, 0], self.normalize(slice_stack[:, 0], pos)
        
    def normalize(self, inpt, pos):
        mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{self.exptime}_pos{pos}'][['mean', 'std']].values[0]
        inpt = (inpt - float(mean)) / float(std)
        return inpt

    def getTestImg(self, identifier, x_start=None, y_start=None, patch_size=None):
        """
        Load test image with corresponding 200ms scan
        """
        pos, angle_index = identifier.split('_')
        df_row = self.df[self.df['identifier'] == identifier]
        shape_x, shape_y = df_row["shape_x"].values[0], df_row["shape_y"].values[0]

        if patch_size == None:
            patch_size = self.patch_size

        if patch_size == "max":
            patch_size = (int(shape_y), int(shape_x))
            
        if x_start is None:
            x_start = random.randint(0, shape_x - patch_size[1])
        if y_start is None:
            y_start = random.randint(0, shape_y - patch_size[0])

        filename_inp = f'{self.path}projStitched_{self.exptime}_{pos}.npy'
        filename_target = f'{self.path}projStitched_200ms_{pos}.npy'


        angle_index = int(angle_index)
        if angle_index + self.n_proj - 1 > 1799:
            angle_index = 1799 - self.n_proj + 1
        inpt = np.load(filename_inp, mmap_mode='c')[angle_index:angle_index + self.n_proj, :, y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
        target = np.load(filename_target, mmap_mode='c')[angle_index:angle_index + self.n_proj, :, y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
        
        inpt, target = torch.from_numpy(inpt).squeeze().float(), torch.from_numpy(target).squeeze().float()
        inpt, target  = self.normalize(inpt, pos), self.normalize(target, pos)
        
        return inpt, target

class ProjDataset(torch.utils.data.Dataset):
    #Load projection data only, with option for sparse-sampling

    def __init__(self, path_proj, path_reco, df_path, df_stats_path, exptime, n_slices, skip=1, n_slicesPR=5, sparseSampling=1):
        
        self.df = pd.read_csv(df_path)
        self.positions = [int(elem) for elem in self.df['position'].values]
        if exptime=='15ms':
            self.positions = [int(elem) for elem in list(self.positions) if 0 != elem]
        self.identifier_list = [f'pos:{pos}_slIndex:{slice_index}' for pos in self.positions for slice_index in range(0, 2150)][::skip]

        self.path_proj = path_proj
        self.path_reco = path_reco
        self.exptime = exptime
        self.df_stats = pd.read_csv(df_stats_path)
        self.n_slices = n_slices
        self.total_n_slices = 2149
        self.n_slicesPR = n_slicesPR
        self.sparseSampling = sparseSampling

    def __len__(self):
        return len(self.identifier_list)

    def __getitem__(self, idx):
        identifier = self.identifier_list[idx % len(self.identifier_list)]
        ids = identifier.split('_')
       
        pos, slice_index = int(ids[0].split(':')[1]), int(ids[1].split(':')[1])
        #print('get proj stack')
        proj_stack = self.load(pos, slice_index)

        return proj_stack, pos, self.exptime

    def load(self, pos, slice_index, exptime=None):
        total_slices = self.total_n_slices

        n_slices = self.n_slicesPR

        # Adjust slice index if needed
        if slice_index + n_slices - 1 > total_slices:
            slice_index = total_slices - n_slices + 1

        # Create slice selection once
        proj_selection = (
            slice(None),
            slice(None),
            slice(slice_index, slice_index + n_slices),
            slice(None)
        )
        if exptime is None:
            exptime = self.exptime

        proj_stack = np.load(f'{self.path_proj}/projStitched_{exptime}_pos{pos}.npy', mmap_mode='c')[proj_selection]
        proj_stack = proj_stack.astype('float32')

        # Ensure the last dimension of proj_stack is even
        if proj_stack.shape[-1] % 2 != 0:
            proj_stack = proj_stack[..., :-1]

        # If sparse-sampling, just just return the projections and reconstruct the sparse-view reconstruction on the fly
        if self.sparseSampling > 1:
            proj_stack = proj_stack[::self.sparseSampling]

        return proj_stack[:, 0]
        