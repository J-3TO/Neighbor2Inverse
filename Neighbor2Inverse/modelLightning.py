import lightning as pl
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lightning.pytorch.callbacks import Callback
import yaml
from torchvision.utils import save_image
from torch_radon import Radon
import pandas as pd
from utilForwardProp import propTIE_torch

torch.autograd.set_detect_anomaly(True)
operation_seed_counter = 0

def get_generator(device="cuda"):
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    device = img.device
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64, device=img.device)
    
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(device=device),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

class Neighbor2NeighborModule(pl.LightningModule):
    def __init__(self, 
                 network, 
                 Lambda1, 
                 Lambda2, 
                 increase_ratio, 
                 n_epoch, 
                 lr, 
                 loss_type='mse', 
                 optimizer_algo='Adam', 
                 scheduler_algo='MultiStepLR', 
                 optimizer_params={}, 
                 scheduler_params={}, 
                gamma=0.5):
        
        super().__init__()
        self.network = network
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.increase_ratio = increase_ratio
        self.n_epoch = n_epoch
        self.lr = lr
        self.loss_type = loss_type
        self.optimizer_algo = optimizer_algo
        self.optimizer_params = optimizer_params
        self.scheduler_algo = scheduler_algo
        self.scheduler_params = scheduler_params
        self.gamma = gamma
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        noisy = batch
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        
        with torch.no_grad():
            noisy_denoised = self(noisy)
            
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = self(noisy_sub1)
        noisy_target = noisy_sub2
        
        Lambda = self.current_epoch / self.n_epoch * self.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        if self.loss_type == 'mse':
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            
        if self.loss_type == 'l1':
            loss1 = torch.mean(torch.abs(diff))
            loss2 = Lambda * torch.mean(torch.abs(diff - exp_diff))
            
        loss = self.Lambda1 * loss1 + self.Lambda2 * loss2
        
        self.log('train_loss1', loss1,  on_epoch=True, sync_dist=True)
        self.log('train_loss2', loss2,  on_epoch=True, sync_dist=True)
        self.log('train_loss', loss,  on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
        
    def get_train_images(self, batch, batch_idx):
        noisy = batch
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        
        with torch.no_grad():
            noisy_denoised = self(noisy)
            
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = self(noisy_sub1)
        noisy_target = noisy_sub2
        
        Lambda = 1
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        if self.loss_type == 'mse':
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            
        if self.loss_type == 'l1':
            loss1 = torch.mean(torch.abs(diff))
            loss2 = Lambda * torch.mean(torch.abs(diff - exp_diff))
            
        loss = self.Lambda1 * loss1 + self.Lambda2 * loss2

        return loss1, loss2, loss, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised

    def validation_step(self, batch, batch_idx):
        noisy = batch
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        
        with torch.no_grad():
            noisy_denoised = self(noisy)
            
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = self(noisy_sub1)
        noisy_target = noisy_sub2
        
        Lambda = self.current_epoch / self.n_epoch * self.increase_ratio
        diff = noisy_output - noisy_target
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

        if self.loss_type == 'mse':
            loss1 = torch.mean(diff**2)
            loss2 = torch.mean((diff - exp_diff)**2)
            
        if self.loss_type == 'l1':
            loss1 = torch.mean(torch.abs(diff))
            loss2 = torch.mean(torch.abs(diff - exp_diff))
            
        loss = self.Lambda1 * loss1 + self.Lambda2 * loss2
        
        self.log('val_loss1', loss1,  on_epoch=True, sync_dist=True)
        self.log('val_loss2', loss2,  on_epoch=True, sync_dist=True)
        self.log('val_loss', loss,  on_epoch=True, sync_dist=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        if self.optimizer_algo == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        if self.optimizer_algo == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **self.optimizer_params)

        if self.scheduler_algo == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "MultiStepLR":
             scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(20 * self.n_epoch/100) - 1,
                    int(40 * self.n_epoch/100) - 1,
                    int(60 * self.n_epoch/100) - 1,
                    int(80 * self.n_epoch/100) - 1
                ],
                 **self.scheduler_params
            )
            
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'}
        

class Neighbor2InverseSlice(pl.LightningModule):
    def __init__(self, 
                 network, 
                 Lambda1, 
                 Lambda2, 
                 increase_ratio, 
                 n_epoch, 
                 lr, 
                 loss_type='mse', 
                 optimizer_algo='Adam', 
                 scheduler_algo='MultiStepLR', 
                 optimizer_params={}, 
                 scheduler_params={}, 
                gamma=0.5,
                regularizer=True,
                df_stats_path = None,
                subsampling = 'projection',
                doPhaseRetrieval = False,
                PR_params = None,
                n_slices = 1,
                n_slicesPR = None,
                sparseSampling = 1,
                dataFidelity = False

                ):
        
        super().__init__()
        self.network = network
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.increase_ratio = increase_ratio
        self.n_epoch = n_epoch
        self.lr = lr
        self.loss_type = loss_type
        self.optimizer_algo = optimizer_algo
        self.optimizer_params = optimizer_params
        self.scheduler_algo = scheduler_algo
        self.scheduler_params = scheduler_params
        self.gamma = gamma
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.regularizer = regularizer
        self.df_stats = pd.read_csv(df_stats_path)
        self.subsampling = subsampling
        self.n_slices = n_slices
        self.n_slicesPR = n_slicesPR

        if self.subsampling == 'sinogram':
            self.factor = 1 
        else:
            self.factor = 2
        self.doPhaseRetrieval = doPhaseRetrieval
        if self.doPhaseRetrieval:
            self.batchsizePR = PR_params['batchsizePR']
            self.beta = PR_params['beta']
            self.pixel_size = PR_params['pixel_size']
            self.wavelength = (12.398424 * 10**(-7)) / PR_params['energy']  # in cm
            self.mu = 4 * np.pi * self.beta / self.wavelength      
            self.sigma = PR_params['delta'] / self.mu * PR_params['z']

        self.sparseSampling = sparseSampling

    def forward(self, x):
        return self.network(x)
        
    def training_step(self, batch, batch_idx):

        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        if len(batch) == 3:
            noisy, pos, exptime = batch
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
        else:
            noisy, reco, pos, exptime = batch

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        del noisy_sub1, noisy_sub2
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        del proj_sub_stack
        torch.cuda.empty_cache()

        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        del sin_stack_phase
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        del reco_sub
        torch.cuda.empty_cache()
        
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        del reco_sub1
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        del noisy_inpt, reco_sub2, pad
        torch.cuda.empty_cache()

        if self.regularizer:
            #compute L_reg as in the paper
            #only tested for proj subsampling, for sino subsampling, this part probably needs some adjustments
            Lambda = self.current_epoch / self.n_epoch * self.increase_ratio     
            with torch.no_grad():
                if self.sparseSampling > 1:
                    reco = noisyPR[:, :, self.n_slicesPR//(self.factor*2)-1:self.n_slicesPR//(self.factor*2)].swapaxes(1, 2)
                    reco = self.normalize(self.reconstruct(reco, angles=angles), pos, exptime) 
                    del noisyPR
                    # Split and process original reconstructions

                if self.subsampling == 'sinogram':
                    reco_pad, pad = pad_to_divisible(reco, 32)
                    reco_denoised = unpad_from_divisible(self(reco_pad), pad)
                    reco_denoised = self.re_normalize(reco_denoised, pos, exptime)
                    del reco_pad, pad
                    torch.cuda.empty_cache()
                
                else:
                    reco1, reco2 = reco[:, :self.n_slices], reco[:, self.n_slices:]
                    del reco
                    reco1_pad, pad1 = pad_to_divisible(reco1, 32)
                    reco_denoised1 = unpad_from_divisible(self(reco1_pad), pad1)
                    del reco1_pad, pad1, reco1
                    torch.cuda.empty_cache()

                    reco2_pad, pad2 = pad_to_divisible(reco2, 32)
                    reco_denoised2 = unpad_from_divisible(self(reco2_pad), pad2)
                    del reco2_pad, pad2, reco2
                    torch.cuda.empty_cache()

                    # Concatenate denoised reconstructions
                    reco_denoised = self.re_normalize(
                        torch.concat((reco_denoised1, reco_denoised2), axis=0), pos, exptime
                    )
                    del reco_denoised1, reco_denoised2
                    torch.cuda.empty_cache()
                
                # Forward projection of denoised reconstruction
                forward_denoised = self.forward_proj(reco_denoised, angles=angles)
                del reco_denoised
                torch.cuda.empty_cache()
                forward_denoised = forward_denoised.swapaxes(0, 1)

            # Generate subimages for forward projection
            if self.subsampling == 'sinogram':
                forward_denoised = forward_denoised.swapaxes(1, 2)

            if self.doPhaseRetrieval and self.subsampling == 'projection':
                difference = self.n_slicesPR - self.n_slices*self.factor

                j = forward_denoised.shape[-1] * (difference // 2)
                k = forward_denoised.shape[-1] * (difference // 2 + self.n_slices*self.factor)
                mask1, mask2= mask1[j:k], mask2[j:k]
    
            # Generate subsampled forward projected images
            forward_denoised_sub1 = generate_subimages(forward_denoised, mask1)
            forward_denoised_sub2 = generate_subimages(forward_denoised, mask2)
            del forward_denoised, mask1, mask2
            torch.cuda.empty_cache()

            if self.subsampling == 'sinogram':
                forward_denoised_sub1 = forward_denoised_sub1.swapaxes(1, 2)
                forward_denoised_sub2 = forward_denoised_sub2.swapaxes(1, 2)
                
            proj_forward_sub_stack = torch.concat((forward_denoised_sub1, forward_denoised_sub2), axis=-2)
            del forward_denoised_sub1, forward_denoised_sub2
            torch.cuda.empty_cache()
            
            sin_stack_phase = proj_forward_sub_stack.swapaxes(1, 2)
            del proj_forward_sub_stack
            torch.cuda.empty_cache()

            # Backward projection
            backward_denoised_sub = self.normalize(self.reconstruct(sin_stack_phase, angles=angles_backproj) / (self.factor), pos, exptime)
            del sin_stack_phase
            torch.cuda.empty_cache()
            
            backward_denoised_sub1, backward_denoised_sub2 = (
                backward_denoised_sub[:, :self.n_slices],
                backward_denoised_sub[:, self.n_slices:],
            )
            del backward_denoised_sub
            torch.cuda.empty_cache()
            
            # Compute losses
            diff = noisy_output - noisy_target
            exp_diff = backward_denoised_sub1 - backward_denoised_sub2
            del backward_denoised_sub1, backward_denoised_sub2
            torch.cuda.empty_cache()
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            del diff, exp_diff
            torch.cuda.empty_cache()
            loss = self.Lambda1 * loss1 + self.Lambda2 * loss2

            # Log losses
            self.log('train_loss1', loss1, on_epoch=True, sync_dist=True)
            self.log('train_loss2', loss2, on_epoch=True, sync_dist=True)
            del loss1, loss2
            torch.cuda.empty_cache()

        else:
            loss = self.loss(noisy_output, noisy_target)

        # Log final loss
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        if len(batch) == 3:
            noisy, pos, exptime = batch
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
        else:
            noisy, reco, pos, exptime = batch

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        del noisy_sub1, noisy_sub2
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        del proj_sub_stack
        torch.cuda.empty_cache()

        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        del sin_stack_phase
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        del reco_sub
        torch.cuda.empty_cache()
        
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        del reco_sub1
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        del noisy_inpt, reco_sub2, pad
        torch.cuda.empty_cache()

        if self.regularizer:
            #compute L_reg as in the paper
            #only tested for proj subsampling, for sino subsampling, this part probably needs some adjustments
            Lambda = 1 # No need to compute this while debugging
            with torch.no_grad():
                if self.sparseSampling > 1:
                    reco = noisyPR[:, :, self.n_slicesPR//(self.factor*2)-1:self.n_slicesPR//(self.factor*2)].swapaxes(1, 2)
                    reco = self.normalize(self.reconstruct(reco, angles=angles), pos, exptime) 
                    del noisyPR
                    # Split and process original reconstructions

                if self.subsampling == 'sinogram':
                    reco_pad, pad = pad_to_divisible(reco, 32)
                    reco_denoised = unpad_from_divisible(self(reco_pad), pad)
                    reco_denoised = self.re_normalize(reco_denoised, pos, exptime)
                    del reco_pad, pad
                    torch.cuda.empty_cache()
                
                else:
                    reco1, reco2 = reco[:, :self.n_slices], reco[:, self.n_slices:]
                    del reco
                    reco1_pad, pad1 = pad_to_divisible(reco1, 32)
                    reco_denoised1 = unpad_from_divisible(self(reco1_pad), pad1)
                    del reco1_pad, pad1, reco1
                    torch.cuda.empty_cache()

                    reco2_pad, pad2 = pad_to_divisible(reco2, 32)
                    reco_denoised2 = unpad_from_divisible(self(reco2_pad), pad2)
                    del reco2_pad, pad2, reco2
                    torch.cuda.empty_cache()

                    # Concatenate denoised reconstructions
                    reco_denoised = self.re_normalize(
                        torch.concat((reco_denoised1, reco_denoised2), axis=0), pos, exptime
                    )
                    del reco_denoised1, reco_denoised2
                    torch.cuda.empty_cache()
                
                # Forward projection of denoised reconstruction
                forward_denoised = self.forward_proj(reco_denoised, angles=angles)
                del reco_denoised
                torch.cuda.empty_cache()
                forward_denoised = forward_denoised.swapaxes(0, 1)

            # Generate subimages for forward projection
            if self.subsampling == 'sinogram':
                forward_denoised = forward_denoised.swapaxes(1, 2)

            if self.doPhaseRetrieval and self.subsampling == 'projection':
                difference = self.n_slicesPR - self.n_slices*self.factor

                j = forward_denoised.shape[-1] * (difference // 2)
                k = forward_denoised.shape[-1] * (difference // 2 + self.n_slices*self.factor)
                mask1, mask2= mask1[j:k], mask2[j:k]
    
            # Generate subsampled forward projected images
            forward_denoised_sub1 = generate_subimages(forward_denoised, mask1)
            forward_denoised_sub2 = generate_subimages(forward_denoised, mask2)
            del forward_denoised, mask1, mask2
            torch.cuda.empty_cache()

            if self.subsampling == 'sinogram':
                forward_denoised_sub1 = forward_denoised_sub1.swapaxes(1, 2)
                forward_denoised_sub2 = forward_denoised_sub2.swapaxes(1, 2)
                
            proj_forward_sub_stack = torch.concat((forward_denoised_sub1, forward_denoised_sub2), axis=-2)
            del forward_denoised_sub1, forward_denoised_sub2
            torch.cuda.empty_cache()
            
            sin_stack_phase = proj_forward_sub_stack.swapaxes(1, 2)
            del proj_forward_sub_stack
            torch.cuda.empty_cache()

            # Backward projection
            backward_denoised_sub = self.normalize(self.reconstruct(sin_stack_phase, angles=angles_backproj) / (self.factor), pos, exptime)
            del sin_stack_phase
            torch.cuda.empty_cache()
            
            backward_denoised_sub1, backward_denoised_sub2 = (
                backward_denoised_sub[:, :self.n_slices],
                backward_denoised_sub[:, self.n_slices:],
            )
            del backward_denoised_sub
            torch.cuda.empty_cache()
            
            # Compute losses
            diff = noisy_output - noisy_target
            exp_diff = backward_denoised_sub1 - backward_denoised_sub2
            del backward_denoised_sub1, backward_denoised_sub2
            torch.cuda.empty_cache()
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            del diff, exp_diff
            torch.cuda.empty_cache()
            loss = self.Lambda1 * loss1 + self.Lambda2 * loss2

            # Log losses
            self.log('val_loss1', loss1, on_epoch=True, sync_dist=True)
            self.log('val_loss2', loss2, on_epoch=True, sync_dist=True)
            del loss1, loss2
            torch.cuda.empty_cache()

        else:
            loss = self.loss(noisy_output, noisy_target)

        # Log final loss
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def get_train_images(self, batch, batch_idx):
        '''
        Function for debugging. Does the same things as the training_step/validation_step function, but returns the images
        '''
        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        print('batch length:', len(batch))
        if len(batch) == 3:
            noisy, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'pos:', pos, 'exptime:', exptime)
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'noisyPR shape:', noisyPR.shape, 'pos:', pos, 'exptime:', exptime)
        else:
            noisy, reco, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'reco shape:', reco.shape, 'pos:', pos, 'exptime:', exptime)

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]
        print(n_angles, angles.shape)

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        print('noisy_sub1 shape:', noisy_sub1.shape, 'noisy_sub2 shape:', noisy_sub2.shape)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            print(middle_slice)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        print('noisy_sub1 pr shape:', noisy_sub1.shape, 'noisy_sub2 pr shape:', noisy_sub2.shape)

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        torch.cuda.empty_cache()


        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        torch.cuda.empty_cache()
        print('reco_sub shape', reco_sub.shape, 'reco_sub1 shape:', reco_sub1.shape, 'reco_sub2 shape:', reco_sub2.shape)
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        torch.cuda.empty_cache()

        if self.regularizer:
            #compute L_reg as in the paper
            #only tested for proj subsampling, for sino subsampling, this part probably needs some adjustments
            Lambda = 1 # No need to compute this while debugging
            with torch.no_grad():
                if self.sparseSampling > 1:
                    print("sparse recon")
                    reco = noisyPR[:, :, self.n_slicesPR//(self.factor*2)-1:self.n_slicesPR//(self.factor*2)].swapaxes(1, 2)
                    reco = self.normalize(self.reconstruct(reco, angles=angles), pos, exptime) 
                    # Split and process original reconstructions

                if self.subsampling == 'sinogram':
                    reco_pad, pad = pad_to_divisible(reco, 32)
                    reco_denoised = unpad_from_divisible(self(reco_pad), pad)
                    reco_denoised = self.re_normalize(reco_denoised, pos, exptime)
                    torch.cuda.empty_cache()
                
                else:
                    print('reco shape', reco.shape)
                    reco1, reco2 = reco[:, :self.n_slices], reco[:, self.n_slices:]
                    reco1_pad, pad1 = pad_to_divisible(reco1, 32)
                    reco_denoised1 = unpad_from_divisible(self(reco1_pad), pad1)
                    torch.cuda.empty_cache()

                    reco2_pad, pad2 = pad_to_divisible(reco2, 32)
                    reco_denoised2 = unpad_from_divisible(self(reco2_pad), pad2)
                    torch.cuda.empty_cache()

                    # Concatenate denoised reconstructions
                    reco_denoised = self.re_normalize(
                        torch.concat((reco_denoised1, reco_denoised2), axis=0), pos, exptime
                    )
                    torch.cuda.empty_cache()
                
                # Forward projection of denoised reconstruction
                forward_denoised = self.forward_proj(reco_denoised, angles=angles)
                torch.cuda.empty_cache()
                forward_denoised = forward_denoised.swapaxes(0, 1)

            # Generate subimages for forward projection
            if self.subsampling == 'sinogram':
                forward_denoised = forward_denoised.swapaxes(1, 2)

            print('Before adjusting: mask1 shape:', mask1.shape, 'mask2 shape:', mask2.shape)
            if self.doPhaseRetrieval and self.subsampling == 'projection':
                difference = self.n_slicesPR - self.n_slices*self.factor

                j = forward_denoised.shape[-1] * (difference // 2)
                k = forward_denoised.shape[-1] * (difference // 2 + self.n_slices*self.factor)
                print("diff", difference, j, k, "j_", (difference // 2), (difference // 2 + self.n_slices*self.factor))
                mask1, mask2= mask1[j:k], mask2[j:k]
            print('After adjusting: mask1 shape:', mask1.shape, 'mask2 shape:', mask2.shape)
   
            # Generate subsampled forward projected images
            forward_denoised_sub1 = generate_subimages(forward_denoised, mask1)
            forward_denoised_sub2 = generate_subimages(forward_denoised, mask2)
            torch.cuda.empty_cache()

            if self.subsampling == 'sinogram':
                forward_denoised_sub1 = forward_denoised_sub1.swapaxes(1, 2)
                forward_denoised_sub2 = forward_denoised_sub2.swapaxes(1, 2)
            print("concatenate")
            proj_forward_sub_stack = torch.concat((forward_denoised_sub1, forward_denoised_sub2), axis=-2)
            torch.cuda.empty_cache()
            print("make sinograms")
            sin_stack_phase = proj_forward_sub_stack.swapaxes(1, 2)
            torch.cuda.empty_cache()

            # Backward projection
            print("reconstruct sinogram shape:", sin_stack_phase.shape)
            backward_denoised_sub = self.normalize(self.reconstruct(sin_stack_phase, angles=angles_backproj) / (self.factor), pos, exptime)
            torch.cuda.empty_cache()
            print("split again. Recon shape:", backward_denoised_sub.shape)
            backward_denoised_sub1, backward_denoised_sub2 = (
                backward_denoised_sub[:, :self.n_slices],
                backward_denoised_sub[:, self.n_slices:],
            )
            torch.cuda.empty_cache()
            print('backward_denoised_sub1 shape:', backward_denoised_sub1.shape, 'backward_denoised_sub2 shape:', backward_denoised_sub2.shape)
            # Compute losses
            diff = noisy_output - noisy_target
            exp_diff = backward_denoised_sub1 - backward_denoised_sub2
            torch.cuda.empty_cache()
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            torch.cuda.empty_cache()
            loss = self.Lambda1 * loss1 + self.Lambda2 * loss2

            # Log losses
            self.log('val_loss1', loss1, on_epoch=True, sync_dist=True)
            self.log('val_loss2', loss2, on_epoch=True, sync_dist=True)
            torch.cuda.empty_cache()
            return noisy, reco, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, reco_denoised, forward_denoised_sub1, forward_denoised_sub2, backward_denoised_sub1, backward_denoised_sub2, loss1, loss2, loss

        else:
            loss = self.loss(noisy_output, noisy_target)
            if len(batch) == 4:
                return noisy, reco, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, loss
            else:
                return noisy, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, loss

        # Log final loss
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_algo == "Adam":
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, **self.optimizer_params)
        if self.optimizer_algo == "AdamW":
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, **self.optimizer_params)
        print('optimizer', optimizer)
        if self.scheduler_algo == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "MultiStepLR":
             scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(20 * self.n_epoch/100) - 1,
                    int(40 * self.n_epoch/100) - 1,
                    int(60 * self.n_epoch/100) - 1,
                    int(80 * self.n_epoch/100) - 1
                ],
                 **self.scheduler_params
            )
            
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'}

    def reconstruct(self, sin_stack_phase, crop=True, angles=None, filter="ram-lak"):
            """
            Reconstructs input sinogram stack using Filtered Back Projection (FBP).
            
            This function performs FBP reconstruction on a stack of sinograms. It applies
            padding to handle edge effects and uses the Ram-Lak filter for reconstruction.
            
            Args:
                sin_stack_phase (torch.Tensor): Stack of phase-retrieved sinograms [num_slices, n_angles, det_count]
                crop (bool): Whether to crop the reconstructed image to original size. Defaults to True.
                angles (torch.Tensor): Angular positions in radians. Must match n_angles dimension of input.
                
            Returns:
                torch.Tensor: Stack of reconstructed slices [num_slices, height, width]
            
            Notes:
                - Uses the Ram-Lak filter for FBP reconstruction
                - Applies padding with cosine fade to reduce edge artifacts
                - Final image is cropped to match input dimensions if crop=True
            """
            # Store original size for final cropping
            #print('move tensor to device')
            final_size = sin_stack_phase.shape[-1]
            #print('sin stack phase shape', sin_stack_phase.shape)
            # Apply padding to handle edge effects
            npad = final_size // 2
            det_count = final_size + 2*npad
            image_size = det_count
            padded_sinogram = padding_width_only(sin_stack_phase, npad)
            
            # Create Radon transform operator
            radon = Radon(
                image_size, 
                angles, 
                det_spacing=1,  # Detector pixel spacing
                clip_to_circle=True,  # Restrict reconstruction to circular field of view
                det_count=det_count
            )
            
            # Apply Ram-Lak filter to padded sinogram
            if filter == None:
                filtered_sinogram = padded_sinogram
            else:
                filtered_sinogram = radon.filter_sinogram(padded_sinogram, filter_name=filter)
            
            # Apply backprojection step of FBP
            fbp_filtered = radon.backprojection(filtered_sinogram)

            if crop:
                fbp_filtered = crop_and_mask(fbp_filtered, crop_size=(final_size, final_size))

            return fbp_filtered

    def forward_proj(self, reco_stack, angles=None):
                """
                
                Forward Projects reconstruction into sinogram space.
                
                
                Args:
                    reco_stack_phase (torch.Tensor): Stack of reconstructions
                    angles (torch.Tensor): Angular positions in radians. 
                    
                Returns:
                    torch.Tensor: Stack of forward projected sinograms 
                
                """

                image_size = reco_stack.shape[-1]
                
                radon = Radon(
                    image_size, 
                    angles, 
                    det_spacing=1,  # Detector pixel spacing
                    clip_to_circle=True,  # Restrict reconstruction to circular field of view
                    det_count=image_size
                )
                
                sinogram = radon.forward(reco_stack)

                return sinogram.swapaxes(0, 2)
    

    def pad_to_divisible(self, image, divisor):
        """
        Pads a tensor so that its height and width (last two dimensions) are divisible by a specified number.

        Args:
            image (torch.Tensor): Input tensor of shape (..., H, W).
            divisor (int): The number to which height and width should be divisible.

        Returns:
            torch.Tensor: Padded tensor.
            tuple: A tuple containing the padding before and after for height and width.
        """
        height, width = image.shape[-2:]

        pad_height = (divisor - (height % divisor)) % divisor
        pad_width = (divisor - (width % divisor)) % divisor

        pad_before_height = pad_height // 2
        pad_after_height = pad_height - pad_before_height

        pad_before_width = pad_width // 2
        pad_after_width = pad_width - pad_before_width

        padding = (pad_before_width, pad_after_width, pad_before_height, pad_after_height)  # Left, Right, Top, Bottom
        padded_image = F.pad(image, padding, mode='reflect')

        return padded_image, (pad_before_height, pad_after_height, pad_before_width, pad_after_width)
    
    def unpad_from_divisible(self, padded_image, original_size):
        """
        Unpads a tensor to its original size after padding.

        Args:
            padded_image (torch.Tensor): Padded tensor of shape (..., H, W).
            original_size (tuple): A tuple containing the padding before and after for height and width.

        Returns:
            torch.Tensor: Unpadded tensor.
        """
        pad_before_height, pad_after_height, pad_before_width, pad_after_width = original_size
        unpadded_image = padded_image[..., pad_before_height:-pad_after_height, pad_before_width:-pad_after_width]
        return unpadded_image

    def normalize(self, inpt, pos, exptime):
        mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
        inpt = (inpt - float(mean)) / float(std)
        return inpt

    def re_normalize(self, inpt, pos, exptime):
        mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
        inpt = inpt*float(std) + float(mean)
        return inpt
    
    # Uncomment the following method to print gradient norms after each backward pass for debugging
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name}: {param.grad.norm()}")
    #         else:
    #             print(f"{name}: No gradient")


class Neighbor2InverseDataFidelity(pl.LightningModule):
    def __init__(self, 
                 network, 
                 LambdaN2N,
                 LambdaFidelity, 
                 n_epoch, 
                 lr, 
                 loss_type='mse', 
                 optimizer_algo='Adam', 
                 scheduler_algo='MultiStepLR', 
                 optimizer_params={}, 
                 scheduler_params={}, 
                df_stats_path = None,
                subsampling = 'projection',
                doPhaseRetrieval = False,
                PR_params = None,
                n_slices = 1,
                n_slicesPR = None,
                sparseSampling = 1,
                forwardProp = True,
                dataFidelity = True,
                virtualSinogram = True,
                ):
        
        super().__init__()
        self.network = network
        self.LambdaN2N = LambdaN2N
        self.LambdaFidelity = LambdaFidelity
        self.n_epoch = n_epoch
        self.lr = lr
        self.loss_type = loss_type
        self.optimizer_algo = optimizer_algo
        self.optimizer_params = optimizer_params
        self.scheduler_algo = scheduler_algo
        self.scheduler_params = scheduler_params
        if loss_type == 'mse':
            self.loss = torch.nn.MSELoss(reduction='mean')
        if loss_type == 'l1':
            self.loss =  torch.nn.L1Loss(reduction='mean')
        self.df_stats = pd.read_csv(df_stats_path)
        self.subsampling = subsampling
        self.n_slices = n_slices
        self.n_slicesPR = n_slicesPR
        self.forwardProp = forwardProp
        self.dataFidelity = dataFidelity
        if self.subsampling == 'sinogram':
            self.factor = 1
        else:
            self.factor = 2
        self.doPhaseRetrieval = doPhaseRetrieval
        self.virtualSinogram = virtualSinogram
        if self.doPhaseRetrieval:

            self.batchsizePR = PR_params['batchsizePR']
            self.beta = PR_params['beta']
            self.pixel_size = PR_params['pixel_size']
            self.wavelength = (12.398424 * 10**(-7)) / PR_params['energy']  # X-ray wavelength (mm)
            self.mu = 4 * np.pi * self.beta / self.wavelength  
            self.delta = PR_params['delta']  
            self.distance = PR_params['z']
            self.sigma = PR_params['delta'] / self.mu * PR_params['z']
            self.energy = PR_params['energy']

        self.sparseSampling = sparseSampling

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        if len (batch) == 3:
            noisy, pos, exptime = batch
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
        else:
            noisy, reco, pos, exptime = batch

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        del mask1, mask2
        
        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        if self.dataFidelity == True and self.virtualSinogram == False:
            # save origSino for later
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_measurement_target = noisy_sub1[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        del noisy_sub1, noisy_sub2
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        del proj_sub_stack
        torch.cuda.empty_cache()

        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        del sin_stack_phase
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        del reco_sub
        torch.cuda.empty_cache()
        
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        del reco_sub1
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        del noisy_inpt, pad
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        torch.cuda.empty_cache()

        lossN2N = self.loss(noisy_output, noisy_target)  
        torch.cuda.empty_cache()

        # Forward projection of denoised reconstruction
        forward_denoised = self.forward_proj(self.re_normalize(noisy_output, pos, exptime), angles=angles)

        #do forward phase prop 
        forward_denoised = padding(forward_denoised, npad_x=1000, npad_y=50)
        delta_map = forward_denoised * self.delta*10**-3
        beta_map = forward_denoised * self.beta*10**-3
        forward_denoised_pp = propTIE_torch(  
            delta=delta_map, 
            beta=beta_map, 
            energy=self.energy, 
            distance=self.distance, 
            px=pixel_size, 
            ind_terms=False, 
            supersample=1, 
            mode="TIE",
            use_float64 = False,
            batch_size = 225
            )
        del delta_map, beta_map
        torch.cuda.empty_cache()
        forward_denoised_pp = forward_denoised_pp[...,  50:-50, 1000:-1000]

        if self.virtualSinogram:
            forward_reco = self.forward_proj(reco_sub2, angles=angles)
            del reco_sub2
            torch.cuda.empty_cache()

            forward_reco = padding(forward_reco, npad_x=1000, npad_y=50)
            delta_map = forward_reco * self.delta*10**-3
            beta_map = forward_reco * self.beta*10**-3
            forward_reco_pp = propTIE_torch(  
                delta=delta_map, 
                beta=beta_map, 
                energy=self.energy, 
                distance=self.distance, 
                px=pixel_size, 
                ind_terms=False, 
                supersample=1, 
                mode="TIE",
                use_float64 = False,
                batch_size = 450
                )
            del delta_map, beta_map, forward_reco
            forward_reco_pp = forward_reco_pp[...,  50:-50, 1000:-1000]

            fidelity_loss = self.loss(forward_denoised_pp, forward_reco_pp)
            del forward_reco_pp
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
            fidelity_loss = self.loss(forward_denoised_pp, noisy_measurement_target)
            del noisy_measurement_target

        del forward_denoised_pp
        loss = self.LambdaN2N * lossN2N + self.LambdaFidelity * fidelity_loss

        # Log losses
        self.log('train_lossN2N', lossN2N, on_epoch=True, sync_dist=True)
        self.log('train_lossFidelity', fidelity_loss, on_epoch=True, sync_dist=True)
        del lossN2N, fidelity_loss, noisy_output, noisy_target
        torch.cuda.empty_cache()
        # Log final loss
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
  
    def validation_step(self, batch, batch_idx):
        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        if len (batch) == 3:
            noisy, pos, exptime = batch
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
        else:
            noisy, reco, pos, exptime = batch

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)
        del mask1, mask2
        
        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        if self.dataFidelity == True and self.virtualSinogram == False:
            # save origSino for later
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_measurement_target = noisy_sub1[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        del noisy_sub1, noisy_sub2
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        del proj_sub_stack
        torch.cuda.empty_cache()

        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        del sin_stack_phase
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        del reco_sub
        torch.cuda.empty_cache()
        
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        del reco_sub1
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        del noisy_inpt, pad
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        torch.cuda.empty_cache()

        lossN2N = self.loss(noisy_output, noisy_target)  
        torch.cuda.empty_cache()

        # Forward projection of denoised reconstruction
        forward_denoised = self.forward_proj(self.re_normalize(noisy_output, pos, exptime), angles=angles)

        #do forward phase prop 
        forward_denoised = padding(forward_denoised, npad_x=1000, npad_y=50)
        delta_map = forward_denoised * self.delta*10**-3
        beta_map = forward_denoised * self.beta*10**-3
        forward_denoised_pp = propTIE_torch(  
            delta=delta_map, 
            beta=beta_map, 
            energy=self.energy, 
            distance=self.distance, 
            px=pixel_size, 
            ind_terms=False, 
            supersample=1, 
            mode="TIE",
            use_float64 = False,
            batch_size = 225
            )
        del delta_map, beta_map
        torch.cuda.empty_cache()
        forward_denoised_pp = forward_denoised_pp[...,  50:-50, 1000:-1000]

        if self.virtualSinogram:
            forward_reco = self.forward_proj(reco_sub2, angles=angles)
            del reco_sub2
            torch.cuda.empty_cache()

            forward_reco = padding(forward_reco, npad_x=1000, npad_y=50)
            delta_map = forward_reco * self.delta*10**-3
            beta_map = forward_reco * self.beta*10**-3
            forward_reco_pp = propTIE_torch(  
                delta=delta_map, 
                beta=beta_map, 
                energy=self.energy, 
                distance=self.distance, 
                px=pixel_size, 
                ind_terms=False, 
                supersample=1, 
                mode="TIE",
                use_float64 = False,
                batch_size = 450
                )
            del delta_map, beta_map, forward_reco
            forward_reco_pp = forward_reco_pp[...,  50:-50, 1000:-1000]

            fidelity_loss = self.loss(forward_denoised_pp, forward_reco_pp)
            del forward_reco_pp
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
            fidelity_loss = self.loss(forward_denoised_pp, noisy_measurement_target)
            del noisy_measurement_target

        del forward_denoised_pp
        loss = self.LambdaN2N * lossN2N + self.LambdaFidelity * fidelity_loss

        # Log losses
        self.log('val_lossN2N', lossN2N, on_epoch=True, sync_dist=True)
        self.log('val_lossFidelity', fidelity_loss, on_epoch=True, sync_dist=True)
        del lossN2N, fidelity_loss, noisy_output, noisy_target
        torch.cuda.empty_cache()
        # Log final loss
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def get_train_images(self, batch, batch_idx):
        '''
        Function for debugging. Does the same things as the training_step function, but returns the images
        '''
        #load precalculated reconstructions for non-sparse sampling approach, reconstruct on the fly for sparse-sampling data
        if len (batch) == 3:
            noisy, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'pos:', pos, 'exptime:', exptime)
        elif len(batch) == 4 and self.sparseSampling > 1:
            noisy, noisyPR, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'noisyPR shape:', noisyPR.shape)
        else:
            noisy, reco, pos, exptime = batch
            print('noisy shape:', noisy.shape, 'reco shape:', reco.shape)

        # Move constants outside the loop
        n_angles = int(noisy.shape[1])
        angles = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]
        print(n_angles, angles.shape)

        # Generate masks and subimages
        if self.subsampling == 'sinogram':
            noisy = noisy.swapaxes(1, 2)  # Inplace operation
        mask1, mask2 = generate_mask_pair(noisy)
        noisy_sub1 = generate_subimages(noisy, mask1)
        noisy_sub2 = generate_subimages(noisy, mask2)

        if self.subsampling == 'sinogram':
            noisy_sub1 = noisy_sub1.swapaxes(1, 2)
            noisy_sub2 = noisy_sub2.swapaxes(1, 2)

        print('noisy_sub1 shape:', noisy_sub1.shape, 'noisy_sub2 shape:', noisy_sub2.shape)

        if self.dataFidelity == True and self.virtualSinogram == False:
            # save origSino for later
            middle_slice = self.n_slicesPR//(self.factor*2)
            print(middle_slice)
            noisy_measurement_target = noisy_sub1[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)

        n_angles = int(noisy_sub1.shape[1])
        angles_backproj = torch.linspace(0, np.pi, n_angles + 1, device=noisy.device)[:-1]

        if self.doPhaseRetrieval:
            if self.subsampling == 'sinogram':
                pixel_size = (self.pixel_size*2, self.pixel_size)
            else:
                pixel_size = self.pixel_size
            # perform phase retrieval on subimages and keep only the central slice
            middle_slice = self.n_slicesPR//(self.factor*2)
            print(middle_slice)
            noisy_sub1 = compute_paganin_batch(noisy_sub1.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            noisy_sub2 = compute_paganin_batch(noisy_sub2.squeeze(), mu=self.mu, sigma=self.sigma, pixel_size=pixel_size, batch_size=self.batchsizePR)[:, :, middle_slice:middle_slice+1].swapaxes(0, 1)
            torch.cuda.empty_cache()

        print('noisy_sub1 pr shape:', noisy_sub1.shape, 'noisy_sub2 pr shape:', noisy_sub2.shape)

        # Concatenate and swap axes for sinogram stack
        proj_sub_stack = torch.concat((noisy_sub1, noisy_sub2), axis=0)
        torch.cuda.empty_cache()
        sin_stack_phase = proj_sub_stack.swapaxes(1, 2)
        torch.cuda.empty_cache()


        reco_sub = self.reconstruct(sin_stack_phase, angles=angles_backproj) / 2 #check out if this factor makes sense for other subsampling methods
        torch.cuda.empty_cache()
        
        reco_sub1, reco_sub2 = reco_sub[:self.n_slices], reco_sub[self.n_slices:]
        torch.cuda.empty_cache()
        print('reco_sub shape', reco_sub.shape, 'reco_sub1 shape:', reco_sub1.shape, 'reco_sub2 shape:', reco_sub2.shape)
        # Compute noisy output and target
        noisy_inpt, pad = pad_to_divisible(reco_sub1, 32)
        torch.cuda.empty_cache()
        noisy_inpt = self.normalize(noisy_inpt, pos, exptime)
        noisy_output = self(noisy_inpt)
        noisy_output = unpad_from_divisible(noisy_output, pad)
        noisy_target = self.normalize(reco_sub2, pos, exptime)
        torch.cuda.empty_cache()

        lossN2N = self.loss(noisy_output, noisy_target)  
        torch.cuda.empty_cache()

        # Forward projection of denoised reconstruction
        #forward_output = self.forward_proj(self.re_normalize(noisy_output, pos, exptime), angles=angles)
        forward_denoised = self.forward_proj(self.re_normalize(noisy_output, pos, exptime), angles=angles)
        print('forward_output shape:', forward_denoised.shape)

        #do forward phase prop 
        #DoTo check out if 1D prop works, fix code
        forward_denoised = padding(forward_denoised, npad_x=1000, npad_y=50)
        print(forward_denoised.shape)
        delta_map = forward_denoised * self.delta*10**-3
        beta_map = forward_denoised * self.beta*10**-3
        forward_denoised_pp = propTIE_torch(  
            delta=delta_map, 
            beta=beta_map, 
            energy=self.energy, 
            distance=self.distance, 
            px=pixel_size, 
            ind_terms=False, 
            supersample=1, 
            mode="TIE",
            use_float64 = False,
            batch_size = 225
            )
        del delta_map, beta_map
        torch.cuda.empty_cache()
        print('forward denoised pp shape', forward_denoised_pp.shape)
        forward_denoised_pp = forward_denoised_pp[...,  50:-50, 1000:-1000]
        print('forward denoised pp shape after removing pad', forward_denoised_pp.shape)

        if self.virtualSinogram:
            print('calculating fidelity loss with virtual')
            forward_reco = self.forward_proj(reco_sub2, angles=angles)
            torch.cuda.empty_cache()
        
        #forward_reco = forward_reco.swapaxes(0, 1)
        #forward_denoised = forward_denoised.swapaxes(0, 1)

            print('forward reco, forward output shape', forward_reco.shape, forward_denoised.shape)

            forward_reco = padding(forward_reco, npad_x=1000, npad_y=50)
            print(forward_reco.shape)
            delta_map = forward_reco * self.delta*10**-3
            beta_map = forward_reco * self.beta*10**-3
            forward_reco_pp = propTIE_torch(  
                delta=delta_map, 
                beta=beta_map, 
                energy=self.energy, 
                distance=self.distance, 
                px=pixel_size, 
                ind_terms=False, 
                supersample=1, 
                mode="TIE",
                use_float64 = False,
                batch_size = 450
                )
            forward_reco_pp = forward_reco_pp[...,  50:-50, 1000:-1000]

            fidelity_loss = self.loss(forward_denoised_pp, forward_reco_pp)
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

            print('data fidelity with raw measured data')
            print('forward_denoised shape', forward_denoised_pp.shape)
            print('noisy_measurement_target shape', noisy_measurement_target.shape)
            fidelity_loss = self.loss(forward_denoised_pp, noisy_measurement_target)

        loss = self.LambdaN2N * lossN2N + self.LambdaFidelity * fidelity_loss

        # Log losses
        self.log('val_lossN2N', lossN2N, on_epoch=True, sync_dist=True)
        self.log('val_lossFidelity', fidelity_loss, on_epoch=True, sync_dist=True)
        torch.cuda.empty_cache()
        if self.virtualSinogram:
            if self.sparseSampling > 1:
                return noisy, noisyPR, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, forward_denoised, forward_reco, forward_denoised_pp, forward_reco_pp, lossN2N, fidelity_loss, loss
            else:
                return noisy, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, forward_denoised, forward_denoised_pp, forward_reco, forward_reco_pp, lossN2N, fidelity_loss, loss
        else:
            if self.sparseSampling > 1:
                return noisy, noisyPR, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, forward_denoised, forward_denoised_pp, noisy_measurement_target, lossN2N, fidelity_loss, loss
            else:
                return noisy, noisy_sub1, noisy_sub2, reco_sub1, reco_sub2, noisy_inpt, noisy_output, noisy_target, forward_denoised, forward_denoised_pp, noisy_measurement_target, lossN2N, fidelity_loss, loss   
        # Log final loss
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_algo == "Adam":
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, **self.optimizer_params)
        if self.optimizer_algo == "AdamW":
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, **self.optimizer_params)
        print('optimizer', optimizer)
        if self.scheduler_algo == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "MultiStepLR":
             scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(20 * self.n_epoch/100) - 1,
                    int(40 * self.n_epoch/100) - 1,
                    int(60 * self.n_epoch/100) - 1,
                    int(80 * self.n_epoch/100) - 1
                ],
                 **self.scheduler_params
            )
            
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'}

    def reconstruct(self, sin_stack_phase, crop=True, angles=None, filter="ram-lak"):
            """
            Reconstructs input sinogram stack using Filtered Back Projection (FBP).
            
            This function performs FBP reconstruction on a stack of sinograms. It applies
            padding to handle edge effects and uses the Ram-Lak filter for reconstruction.
            
            Args:
                sin_stack_phase (torch.Tensor): Stack of phase-retrieved sinograms [num_slices, n_angles, det_count]
                crop (bool): Whether to crop the reconstructed image to original size. Defaults to True.
                angles (torch.Tensor): Angular positions in radians. Must match n_angles dimension of input.
                
            Returns:
                torch.Tensor: Stack of reconstructed slices [num_slices, height, width]
            
            Notes:
                - Uses the Ram-Lak filter for FBP reconstruction
                - Applies padding with cosine fade to reduce edge artifacts
                - Final image is cropped to match input dimensions if crop=True
            """
            # Store original size for final cropping
            #print('move tensor to device')
            final_size = sin_stack_phase.shape[-1]
            #print('sin stack phase shape', sin_stack_phase.shape)
            # Apply padding to handle edge effects
            npad = final_size // 2
            det_count = final_size + 2*npad
            image_size = det_count
            padded_sinogram = padding_width_only(sin_stack_phase, npad)
            
            # Create Radon transform operator
            radon = Radon(
                image_size, 
                angles, 
                det_spacing=1,  # Detector pixel spacing
                clip_to_circle=True,  # Restrict reconstruction to circular field of view
                det_count=det_count
            )
            
            # Apply Ram-Lak filter to padded sinogram
            filtered_sinogram = radon.filter_sinogram(padded_sinogram, filter_name=filter)
            
            # Apply backprojection step of FBP
            fbp_filtered = radon.backprojection(filtered_sinogram)

            if crop:
                fbp_filtered = crop_and_mask(fbp_filtered, crop_size=(final_size, final_size))

            return fbp_filtered

    def forward_proj(self, reco_stack, angles=None):
                """
                
                Forward Projects reconstruction into sinogram space.
                
                
                Args:
                    reco_stack_phase (torch.Tensor): Stack of reconstructions
                    angles (torch.Tensor): Angular positions in radians. 
                    
                Returns:
                    torch.Tensor: Stack of forward projected sinograms 
                
                """

                # Store original size for final cropping
                #print('move tensor to device')
 
                image_size = reco_stack.shape[-1]
                
                # Create Radon transform operator
                radon = Radon(
                    image_size, 
                    angles, 
                    det_spacing=1,  # Detector pixel spacing
                    clip_to_circle=True,  # Restrict reconstruction to circular field of view
                    det_count=image_size
                )
                
                # Apply Ram-Lak filter to padded sinogram
                sinogram = radon.forward(reco_stack)

                return sinogram.swapaxes(0, 2)

    def pad_to_divisible(self, image, divisor):
        """
        Pads a tensor so that its height and width (last two dimensions) are divisible by a specified number.

        Args:
            image (torch.Tensor): Input tensor of shape (..., H, W).
            divisor (int): The number to which height and width should be divisible.

        Returns:
            torch.Tensor: Padded tensor.
            tuple: A tuple containing the padding before and after for height and width.
        """
        height, width = image.shape[-2:]

        pad_height = (divisor - (height % divisor)) % divisor
        pad_width = (divisor - (width % divisor)) % divisor

        pad_before_height = pad_height // 2
        pad_after_height = pad_height - pad_before_height

        pad_before_width = pad_width // 2
        pad_after_width = pad_width - pad_before_width

        padding = (pad_before_width, pad_after_width, pad_before_height, pad_after_height)  # Left, Right, Top, Bottom
        padded_image = F.pad(image, padding, mode='reflect')

        return padded_image, (pad_before_height, pad_after_height, pad_before_width, pad_after_width)
    
    def unpad_from_divisible(self, padded_image, original_size):
        """
        Unpads a tensor to its original size after padding.

        Args:
            padded_image (torch.Tensor): Padded tensor of shape (..., H, W).
            original_size (tuple): A tuple containing the padding before and after for height and width.

        Returns:
            torch.Tensor: Unpadded tensor.
        """
        pad_before_height, pad_after_height, pad_before_width, pad_after_width = original_size
        unpadded_image = padded_image[..., pad_before_height:-pad_after_height, pad_before_width:-pad_after_width]
        return unpadded_image

    def normalize(self, inpt, pos, exptime):
        mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
        inpt = (inpt - float(mean)) / float(std)
        return inpt

    def re_normalize(self, inpt, pos, exptime):
        mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
        inpt = inpt*float(std) + float(mean)
        return inpt
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name}: {param.grad.norm()}")
    #         else:
    #             print(f"{name}: No gradient")

            


def padding_width_only(tensor: torch.Tensor, npad: int) -> torch.Tensor:
    """
    Pad torch tensor only in the width dimension (last dimension) with cosine fade.
    
    This function applies periodic padding with cosine fade to create a smooth transition
    at the edges of the image in the width dimension only. The height dimension remains
    unchanged.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape [batch, height, width] or [batch, channels, height, width]
        npad (int): Number of padding pixels to add on each side of width dimension
        
    Returns:
        torch.Tensor: Padded tensor with smooth transitions at width boundaries
    """
    device = tensor.device
    # Create cosine fade weights for smooth transition
    weight_major = 0.5 + 0.5 * torch.cos(
        torch.linspace(0., torch.pi * 0.5, npad)).to(device)
    weight_minor = 0.5 + 0.5 * torch.cos(
        torch.linspace(torch.pi, torch.pi * 0.5, npad)).to(device)
    
    # Determine input tensor dimensions
    if len(tensor.shape) == 3:  # [batch, height, width]
        # Apply padding only in the width dimension (dim=2)
        ten_pad = torch.nn.functional.pad(tensor, (npad, npad, 0, 0))
        
        # Create smooth transitions at left border
        ten_pad[..., :npad] = \
            torch.flip(weight_major, (0,))[None, None, :] \
            * ten_pad[..., npad][..., None] \
            + torch.flip(weight_minor, (0,))[None, None, :] \
            * ten_pad[..., -npad][..., None]
        
        # Create smooth transitions at right border
        ten_pad[..., -npad:] = \
            weight_major[None, None, :] \
            * ten_pad[..., -npad-1][..., None] \
            + weight_minor[None, None, :] \
            * ten_pad[..., npad][..., None]
            
    elif len(tensor.shape) == 4:  # [batch, channels, height, width]
        # Apply padding only in the width dimension (dim=3)
        ten_pad = torch.nn.functional.pad(tensor, (npad, npad, 0, 0))
        
        # Create smooth transitions at left border
        ten_pad[..., :npad] = \
            torch.flip(weight_major, (0,))[None, None, None, :] \
            * ten_pad[..., npad][..., None] \
            + torch.flip(weight_minor, (0,))[None, None, None, :] \
            * ten_pad[..., -npad][..., None]
        
        # Create smooth transitions at right border
        ten_pad[..., -npad:] = \
            weight_major[None, None, None, :] \
            * ten_pad[..., -npad-1][..., None] \
            + weight_minor[None, None, None, :] \
            * ten_pad[..., npad][..., None]
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}. Expected 3D or 4D tensor.")
        
    return ten_pad


def crop_and_mask(image: torch.Tensor, crop_size: tuple):
    """
    Crops a batched grayscale PyTorch image tensor to a specified size and applies a circular mask.

    :param image: Input image tensor of shape (B, C, H, W).
    :param crop_size: Tuple (crop_width, crop_height).
    :return: Cropped and masked image tensor of shape (B, C, crop_height, crop_width).
    """
    if len(image.shape) == 3:  # [batch, angles, det_count]
        image = image[:, None]
    B, C, H, W = image.shape  # Batch, Channel, Height, Width
    crop_w, crop_h = crop_size

    # Ensure crop size is valid
    crop_w = min(crop_w, W)
    crop_h = min(crop_h, H)

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop size: {crop_size} for image dimensions: {image.shape}")


    # Compute cropping start points (center crop)
    start_x = (W - crop_w) // 2
    start_y = (H - crop_h) // 2

    # Crop the image (batched indexing)
    cropped_image = image[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]  # (B, C, crop_h, crop_w)

    # Create a circular mask (only need to compute it once per crop size)
    device = image.device  # Ensure mask is on the same device
    y = torch.arange(crop_h, device=device)
    x = torch.arange(crop_w, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    center_y, center_x = crop_h // 2, crop_w // 2
    radius = min(center_y, center_x)  # Radius to fit within cropped area

    mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2)  # (crop_h, crop_w)

    # Expand mask to match batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, crop_h, crop_w)
    mask = mask.expand(B, C, -1, -1)  # (B, C, crop_h, crop_w)

    # Apply the mask
    masked_image = cropped_image * mask  # Element-wise multiplication

    return masked_image  # (B, C, crop_h, crop_w)

def paganin_kernel(shape: tuple, sigma: float,
                   pixel_size: float | tuple = 0.172) -> torch.Tensor:
    """
    Compute kernel for Paganin's phase retrieval.

    Args:
        shape (tuple): Projection shape (..., height, width)
        sigma (float): delta / mu * z
        pixel_size (float or tuple, optional): Sampling period. If a single float is provided, it is used for both
                                               dimensions. If a tuple is provided, it should be (pixel_size_u, pixel_size_v).
                                               Defaults to 0.172.

    Returns:
        torch.Tensor: Kernel
    """
    if isinstance(pixel_size, tuple):
        pixel_size_u, pixel_size_v = pixel_size
    else:
        pixel_size_u = pixel_size_v = pixel_size

    u = torch.fft.fftfreq(shape[-1], d=pixel_size_u)
    v = torch.fft.fftfreq(shape[-2], d=pixel_size_v)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    return 1 / (sigma * 4 * torch.pi**2 * (uu**2 + vv**2) + 1)

def convolve(proj: torch.Tensor,
             kernel: torch.Tensor) -> torch.Tensor:
    """
    Compute convolution on batch.

    Args:
        proj (torch.Tensor): projections (..., height, width)
        kernel (torch.Tensor): 2d kernel for convolution

    Returns:
        torch.Tensor: convolved tensor
    """
    device = proj.device
    kernel = kernel.to(device)
    return torch.fft.ifft2(torch.einsum(
        "hijk, jk -> hijk", torch.fft.fft2(proj), kernel)).real

def compute_phase_retrieval(proj: torch.Tensor, sigma: float,
                            pixel_size: float=0.009) -> torch.Tensor:
    """
    Compute Paganin phase retrieval.

    Args:
        proj (torch.Tensor): projections (..., w, h)
        sigma (float): simplified delta / mu * propagation_distance 
        pixel_size (float): pixel size. Defaults to 0.009.

    Returns:
        torch.Tensor: phase retrieved transmission images.
    """
    kernel = paganin_kernel(proj.shape, sigma, pixel_size)
    return convolve(proj, kernel)

def padding(tensor: torch.Tensor, npad_x: int, npad_y: int) -> torch.Tensor:
    """
    Pad torch tensor to be periodic with cosine fade, allowing individual padding for x and y directions.

    Args:
        tensor (torch.Tensor): Input tensor.
        npad_x (int): Padding size for the x (width) dimension.
        npad_y (int): Padding size for the y (height) dimension.

    Returns:
        torch.Tensor: Padded tensor.
    """
    device = tensor.device

    # Create cosine fade weights for x and y directions
    weight_major_x = 0.5 + 0.5 * torch.cos(torch.linspace(0., torch.pi * 0.5, npad_x)).to(device)
    weight_minor_x = 0.5 + 0.5 * torch.cos(torch.linspace(torch.pi, torch.pi * 0.5, npad_x)).to(device)
    weight_major_y = 0.5 + 0.5 * torch.cos(torch.linspace(0., torch.pi * 0.5, npad_y)).to(device)
    weight_minor_y = 0.5 + 0.5 * torch.cos(torch.linspace(torch.pi, torch.pi * 0.5, npad_y)).to(device)

    # Apply padding in both x and y directions
    ten_pad = torch.nn.functional.pad(tensor, (npad_x, npad_x, npad_y, npad_y))

    # Smooth transitions for y (height) dimension
    if len(tensor.shape) == 4:
        ten_pad[..., :npad_y, :] = (
            torch.flip(weight_major_y, (0,))[None, None, :, None] * ten_pad[..., npad_y, :][..., None, :]
            + torch.flip(weight_minor_y, (0,))[None, None, :, None] * ten_pad[..., -npad_y - 1, :][..., None, :]
        )
        ten_pad[..., -npad_y:, :] = (
            weight_major_y[None, None, :, None] * ten_pad[..., -npad_y - 1, :][..., None, :]
            + weight_minor_y[None, None, :, None] * ten_pad[..., npad_y, :][..., None, :]
        )

        # Smooth transitions for x (width) dimension
        ten_pad[..., :, :npad_x] = (
            torch.flip(weight_major_x, (0,))[None, None, None, :] * ten_pad[..., :, npad_x][..., :, None]
            + torch.flip(weight_minor_x, (0,))[None, None, None, :] * ten_pad[..., :, -npad_x - 1][..., :, None]
        )
        ten_pad[..., :, -npad_x:] = (
            weight_major_x[None, None, None, :] * ten_pad[..., :, -npad_x - 1][..., :, None]
            + weight_minor_x[None, None, None, :] * ten_pad[..., :, npad_x][..., :, None]
        )
    else:
        ten_pad[..., :npad_y, :] = (
            torch.flip(weight_major_y, (0,))[None, :, None] * ten_pad[..., npad_y, :][..., None, :]
            + torch.flip(weight_minor_y, (0,))[None, :, None] * ten_pad[..., -npad_y - 1, :][..., None, :]
        )
        ten_pad[..., -npad_y:, :] = (
            weight_major_y[None, :, None] * ten_pad[..., -npad_y - 1, :][..., None, :]
            + weight_minor_y[None, :, None] * ten_pad[..., npad_y, :][..., None, :]
        )

        # Smooth transitions for x (width) dimension
        ten_pad[..., :, :npad_x] = (
            torch.flip(weight_major_x, (0,))[None, None, :] * ten_pad[..., :, npad_x][..., :, None]
            + torch.flip(weight_minor_x, (0,))[None, None, :] * ten_pad[..., :, -npad_x - 1][..., :, None]
        )
        ten_pad[..., :, -npad_x:] = (
            weight_major_x[None, None, :] * ten_pad[..., :, -npad_x - 1][..., :, None]
            + weight_minor_x[None, None, :] * ten_pad[..., :, npad_x][..., :, None]
        )

    return ten_pad

def compute_paganin_batch(proj_stack,
                          mu, 
                          sigma,
                          pixel_size, 
                          batch_size):  
    
    pad_size_x = proj_stack.shape[-1] // 2
    pad_size_y = proj_stack.shape[-2] // 2

    phas_img_list = []
    # Calculate number of batches with ceiling division
    num_batches = (len(proj_stack) + batch_size - 1) // batch_size
    
    for k in range(num_batches):
        # Extract and prepare current batch
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, len(proj_stack))
        
        # Move batch to GPU, convert to float, and add small constant
        batch = proj_stack[start_idx:end_idx]

        if len(batch.shape) == 3:  # [batch, height, width]
                batch_pad = padding(batch, npad_x=pad_size_x, npad_y=pad_size_y)
        else:  # [batch, channels, height, width]
            batch_pad = padding(batch.view(batch.size(0), batch.size(2), batch.size(3)), npad_x=pad_size_x, npad_y=pad_size_y)
        #print(batch_pad.shape)
        # Add channel dimension if needed
        if len(batch_pad.shape) == 3:
            batch_pad = batch_pad[:, None]
            
        # Apply phase retrieval algorithm
        batch_paganin = compute_phase_retrieval(batch_pad, sigma=sigma, pixel_size=pixel_size)
        
        # Remove padding
        batch_paganin = batch_paganin[:, :, pad_size_y:-pad_size_y, pad_size_x:-pad_size_x]
        
        # Calculate absorption projection from phase-retrieved intensity
        batch_paganin_log = -1 / mu * torch.log(batch_paganin)

        phas_img_list.extend(batch_paganin_log)

    return torch.stack(phas_img_list, axis=0)


def pad_to_divisible(image, divisor):
    """
    Pads a tensor so that its height and width (last two dimensions) are divisible by a specified number.

    Args:
        image (torch.Tensor): Input tensor of shape (..., H, W).
        divisor (int): The number to which height and width should be divisible.

    Returns:
        torch.Tensor: Padded tensor.
        tuple: A tuple containing the padding before and after for height and width.
    """
    height, width = image.shape[-2:]

    pad_height = (divisor - (height % divisor)) % divisor
    pad_width = (divisor - (width % divisor)) % divisor

    pad_before_height = pad_height // 2
    pad_after_height = pad_height - pad_before_height

    pad_before_width = pad_width // 2
    pad_after_width = pad_width - pad_before_width

    padding = (pad_before_width, pad_after_width, pad_before_height, pad_after_height)  # Left, Right, Top, Bottom
    padded_image = F.pad(image, padding, mode='reflect')

    return padded_image, (pad_before_height, pad_after_height, pad_before_width, pad_after_width)

def unpad_from_divisible(padded_image, original_size):
    """
    Unpads a tensor to its original size after padding.

    Args:
        padded_image (torch.Tensor): Padded tensor of shape (..., H, W).
        original_size (tuple): A tuple containing the padding before and after for height and width.

    Returns:
        torch.Tensor: Unpadded tensor.
    """
    pad_before_height, pad_after_height, pad_before_width, pad_after_width = original_size
    
    # Handle height dimension
    if pad_after_height == 0:
        height_slice = slice(pad_before_height, None)
    else:
        height_slice = slice(pad_before_height, -pad_after_height)
    
    # Handle width dimension
    if pad_after_width == 0:
        width_slice = slice(pad_before_width, None)
    else:
        width_slice = slice(pad_before_width, -pad_after_width)
        
    unpadded_image = padded_image[..., height_slice, width_slice]
    return unpadded_image

def normalize(self, inpt, pos, exptime):
    mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
    inpt = (inpt - float(mean)) / float(std)
    return inpt

def re_normalize(self, inpt, pos, exptime):
    mean, std = self.df_stats[self.df_stats['filename'] == f'reco_{exptime[0]}_pos{pos[0]}'][['mean', 'std']].values[0]
    inpt = inpt*float(std) + float(mean)
    return inpt


def process_large_image(model, image, patch_size=512, divisor=32, overlap=0.1, batch_size=4):
    """
    Processes a large image in patches to fit the model's input size requirements,
    while ensuring divisibility by a divisor and minimizing edge artifacts with overlap.

    Args:
        image (torch.Tensor): Input image tensor of shape [1, C, H, W].
        model (torch.nn.Module): The CNN model to apply.
        patch_size (int): The size of the patches to divide the image into.
        divisor (int): The number to which height and width of patches should be divisible.
        overlap (float): The amount of overlap between patches (0.0 to 1.0).
        batch_size (int): The batch size to use when processing patches.

    Returns:
        torch.Tensor: The processed image tensor of the same shape as the input.
    """
    # 1. Calculate Patching Parameters
    batch_size_img, channels, height, width = image.shape
    stride = int(patch_size * (1 - overlap))

    # 2. Pad the Image to be Divisible by patch_size
    pad_height = (stride - (height % stride)) % stride
    pad_width = (stride - (width % stride)) % stride
    padded_image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height), mode='reflect')
    padded_height, padded_width = padded_image.shape[-2:]

    # 3. Generate Patches
    patches = padded_image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, channels, patch_size, patch_size)

    # 4. Pad Patches to be Divisible by Divisor
    padded_patches = []
    original_sizes = []
    for patch in patches:
        padded_patch, original_size = pad_to_divisible(patch.unsqueeze(0), divisor)
        padded_patches.append(padded_patch)
        original_sizes.append(original_size)
    padded_patches = torch.cat(padded_patches, dim=0)

    # 5. Process Patches through the Model in Batches
    processed_patches = []

    for i in range(0, len(padded_patches), batch_size):
        batch_patches = padded_patches[i:i + batch_size]
        processed_batch = model(batch_patches)
        processed_patches.append(processed_batch)
    processed_patches = torch.cat(processed_patches, dim=0)

    # 6. Unpad Patches to Original Patch Size
    unpadded_patches = []
    for i, padded_patch in enumerate(processed_patches):
        unpadded_patch = unpad_from_divisible(padded_patch, original_sizes[i])
        unpadded_patches.append(unpadded_patch)
    unpadded_patches = torch.stack(unpadded_patches, dim=0)

    # 7. Reassemble the Image
    patches_height = (padded_height - patch_size) // stride + 1
    patches_width = (padded_width - patch_size) // stride + 1
    processed_image = torch.zeros_like(padded_image)
    weight_map = torch.zeros_like(padded_image)

    for i in range(patches_height):
        for j in range(patches_width):
            x_start = j * stride
            y_start = i * stride
            patch = unpadded_patches[i * patches_width + j].unsqueeze(0)
            processed_image[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += patch
            weight_map[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size] += 1.0  # Add a weight of 1 for each patch

    # 8. Average the Overlapping Regions
    processed_image /= weight_map

    # 9. Crop the Image to the Original Size
    final_image = processed_image[:, :, :height, :width]

    return final_image
