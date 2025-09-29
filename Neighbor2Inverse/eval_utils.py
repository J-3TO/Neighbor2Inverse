import os
import lightning as pl
import torch
import yaml
import sys
from copy import deepcopy
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_images_with_zoom(images, title_list=None, vmin=None, vmax=None, zoom_center=(50, 50), zoom_size=20, axis=True):
    """
    Plots a set of images with zoomed-in patches.
    
    Parameters:
        images (list of 2D arrays): List of grayscale images to plot.
        vmin (float, optional): Minimum intensity value for normalization.
        vmax (float, optional): Maximum intensity value for normalization.
        zoom_center (tuple): (x, y) coordinates for the center of the zoomed-in region.
        zoom_size (int): Half the size of the zoomed-in region.
    """
    num_images = len(images)
    fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
    
    for i, img in enumerate(images):
        img = img.squeeze()
        axes[0, i].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(title_list[i])
        if not axis:
            axes[0, i].axis('off')
        
        # Extract zoomed-in patch
        x, y = zoom_center
        zoom_patch = img[y-zoom_size:y+zoom_size, x-zoom_size:x+zoom_size]
        
        # Draw red rectangle to indicate zoomed region
        rect = plt.Rectangle((x-zoom_size, y-zoom_size), 2*zoom_size, 2*zoom_size, 
                             edgecolor='red', facecolor='none', linewidth=1)
        axes[0, i].add_patch(rect)
        
        # Plot zoomed-in patch
        axes[1, i].imshow(zoom_patch, cmap='gray', vmin=vmin, vmax=vmax)
        if not axis:
           axes[0, i].axis('off')        
           axes[1, i].axis('off')
    
    plt.tight_layout()
