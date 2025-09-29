import lightning as pl
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lightning.pytorch.callbacks import Callback
import yaml
from torchvision.utils import save_image
import pandas as pd

def normalize_to_range(tensor, vmin, vmax):
    """
    Normalize a tensor to the range [vmin, vmax].
    
    Args:
        tensor (torch.Tensor): Input tensor.
        vmin (float): Minimum value for normalization.
        vmax (float): Maximum value for normalization.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    tensor = torch.clamp(tensor, vmin, vmax)  # Clip values outside [vmin, vmax]
    tensor = (tensor - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    return tensor
   
class SavePredictionCallbackSlice(Callback):
    """
    Callback to save model predictions on a validation dataset slice at the end of each epoch.
    """
    def __init__(self, validation_dataset, output_dir="predictions", example_idx=0):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.output_dir = output_dir
        self.example_idx = example_idx
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        pl_module.eval()
        device = pl_module.device

        # Get the example input image
        proj, input_image, pos, exptime = self.validation_dataset.__getitem__(self.example_idx)
        input_image = torch.from_numpy(input_image[:1, 1024:2048, 1024:2048]).to(device).unsqueeze(0)
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        mn, std = input_image.mean().detach().cpu(), input_image.std().mean().detach().cpu()
        vmin=mn-3*std
        vmax=mn+3*std
        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [input_image.squeeze(0).cpu(),  predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()


class SavePredictionCallback(Callback):
    """
    Callback to save model predictions on a specified test image at the end of each epoch.
    """
    def __init__(self, output_dir="predictions", imagepath_inpt="./TestSlices/test_slice_15ms_1800projs.npy", imagepath_target="./TestSlices/test_slice_15ms_900projs.npy", exptime="15ms"):
        super().__init__()
        self.output_dir = output_dir
        self.imagepath_inpt = imagepath_inpt
        self.imagepath_target = imagepath_target
        self.exptime = exptime
        os.makedirs(output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "epoch")
        
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            # Ensure this runs only on the main process
            self.save_current_progress(trainer, pl_module, "start")
        
    def save_current_progress(self, trainer, pl_module, name):
        """
        Called when the epoch ends. Saves a prediction image to the output directory.
        """
        # Ensure model is in evaluation mode
        pl_module.eval()
        device = pl_module.device

        # Check if files exist before loading
        if not os.path.exists(self.imagepath_target):
            print(f"Warning: Target image file not found: {self.imagepath_target}")
        
        if not os.path.exists(self.imagepath_inpt):
            print(f"Warning: Input image file not found: {self.imagepath_inpt}")

        # Get the example input image
        try:
            fullview_image = np.load(self.imagepath_target).astype('float32')
            fullview_image = torch.from_numpy(fullview_image[1524:2548, 3024:4048]).unsqueeze(0).unsqueeze(0)
            input_image = np.load(self.imagepath_inpt).astype('float32')
            input_image = torch.from_numpy(input_image[1524:2548, 3024:4048]).to(device).unsqueeze(0).unsqueeze(0)
        except Exception as e:
            print(f"Warning: Error loading image files: {e}")
            return

        fullview_image = pl_module.normalize(fullview_image, pos=[3], exptime=[str(self.exptime)])
        input_image = pl_module.normalize(input_image, pos=[3], exptime=[str(self.exptime)])
        #print('shape inpt, callback', input_image.shape)  # Add batch dimension
        mn, std = input_image.mean().detach().cpu(), input_image.std().mean().detach().cpu()
        vmin=mn-3*std
        vmax=mn+3*std
        # Predict using the model
        with torch.no_grad():
            predicted_image = pl_module(input_image)

        # Save the input, target, and predicted images
        img_list = [normalize_to_range(img, vmin=vmin, vmax=vmax) for img in [fullview_image.squeeze(0), input_image.squeeze(0).cpu(),  predicted_image.squeeze(0).cpu()]]
        save_image(img_list, os.path.join(self.output_dir, f"{name}_{trainer.current_epoch}_progress.png"))

        # Switch back to training mode
        pl_module.train()

class SaveHyperparametersCallback(Callback):
    def __init__(self, output_dir, file):
        super().__init__()
        self.output_dir = output_dir
        self.file = file

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:  # Ensure this runs only on the main process
            # Save hyperparameter file to the weights directory
            with open(f'{self.output_dir}/trainparams.yml', 'w') as outfile:
                yaml.dump(self.file, outfile, default_flow_style=False)