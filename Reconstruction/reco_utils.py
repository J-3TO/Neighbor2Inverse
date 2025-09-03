import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch_radon import Radon, RadonFanbeam
import torch.fft
import numpy as np
import math

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    return device

def print_stats(ar):
    print("min:", ar.min(), "mean:", ar.mean(), "max:", ar.max(), "std:", ar.std())

def radon_forward(recons, batch_size, n_angles, device='cuda'):
    num_batches = (len(recons) + batch_size - 1) // batch_size
    
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    radon = Radon(recons.shape[-1], angles, det_spacing=1, clip_to_circle=False, det_count=recons.shape[-1])
    
    sino_list = []
    for k in tqdm(range(num_batches)):
        # Extract and prepare current batch
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, len(recons))
        
        # Move batch to GPU and convert to float
        recon = recons[start_idx:end_idx].to(device).float()  
    
        sinogram_forward = radon.forward(recon)
        sino_list.extend(sinogram_forward.detach.cpu())
    
    return torch.stack(sino_list, dim=0)
    
def recon_batch(sin_stack_phase, batch_size=8, n_angles=1800, centershift=0,  size_rivers=7, device='cuda', crop=True, angles=None, dtype_out=torch.float32):
    """
    Reconstructs 3D volume from phase-retrieved sinograms using Filtered Back Projection (FBP)
    in a batched approach.
    
    This function processes sinograms in batches to efficiently reconstruct slices of a 3D volume.
    It applies several preprocessing steps including padding, ring artifact correction, and 
    center shift correction before performing the FBP reconstruction.
    
    Args:
        sin_stack_phase (torch.Tensor): Stack of phase-retrieved sinograms [num_slices, n_angles, det_count]
        batch_size (int): Number of sinograms to process at once
        n_angles (int): Number of projection angles
        centershift (float): Center of rotation shift correction value in pixels
        device (str): Computation device ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Stack of reconstructed slices [num_slices, height, width]
    
    Notes:
        - Uses the Ram-Lak filter for FBP reconstruction
        - Applies ring artifact correction via the 'rivers' function
        - Corrects for center of rotation misalignment
    """
    # Fix variable name inconsistency (proj_stack to sin_stack_phase)
    final_size = sin_stack_phase.shape[-1]
    num_batches = (len(sin_stack_phase) + batch_size - 1) // batch_size

    if angles == None:
        angles = np.linspace(0, np.pi, n_angles, endpoint=False) 
    else:
        print('Using provided angles')
        
        
    # Apply padding to handle edge effects
    npad = sin_stack_phase.shape[-1] // 2
    det_count = sin_stack_phase.shape[-1] + 2*npad
    image_size = det_count

    stack_pad = padding_width_only(sin_stack_phase, npad)
    
    # Create Radon transform operator
    radon = Radon(
        image_size, 
        angles, 
        det_spacing=1,  # Detector pixel spacing
        clip_to_circle=True,  # Don't restrict to circular field of view
        det_count=det_count
    )
    
    fbp_list = []
    for k in tqdm(range(num_batches)):
        # Extract and prepare current batch
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, len(sin_stack_phase))
        
        # Move batch to GPU and convert to float
        sinogram = stack_pad[start_idx:end_idx]
        #print(sinogram.shape)
        sinogram = sinogram.to(device).float()      

        if len(sinogram.shape) == 3:  # [batch, angles, det_count]
            sinogram = sinogram[:, None] # [batch, channels, angles, det_count]
        
        # Apply ring artifact correction
        sinogram = rivers(sinogram, size_rivers)  # 21 is the filter size for ring artifact correction
        
        # Apply center of rotation correction
        #sinogram = correct_center_shift(sinogram, shift=centershift)
        
        # Define reconstruction geometrys

        
        # Apply filtering step of FBP
        filtered_sinogram = radon.filter_sinogram(sinogram, filter_name="ram-lak")
        
        # Apply backprojection step of FBP and transfer result to CPU
        fbp_filtered = radon.backprojection(filtered_sinogram)

        if crop:
            fbp_filtered = crop_and_mask(fbp_filtered, crop_size=(final_size, final_size))
            
        fbp_filtered = fbp_filtered.detach().to("cpu", dtype=dtype_out)
        
        # Add reconstructed slices to result list
        fbp_list.append(fbp_filtered)
    
    # Stack all reconstructed batches into a single tensor
    fbp_list = torch.cat(fbp_list, dim=0)
    return fbp_list

    
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

def torch_dead_correction(im):
    """Correct dead pixels by interpolation for PyTorch tensors
    
    Parameters
    ----------
    im : torch.Tensor
        Image data (sinogram) as torch tensor with shape (batch, channel, height, width)
    
    Returns
    -------
    im : torch.Tensor
        Corrected image data
    """
    # Create a copy to avoid modifying the input
    im = im.clone()
    
    # Handle each batch and channel separately
    for b in range(im.shape[0]):
        for c in range(im.shape[1]):
            # Get current slice
            im_slice = im[b, c]
            
            # Set negative values to zero
            im_slice[im_slice < 0.0] = 0.0
            
            # Flatten for interpolation
            im_f = im_slice.flatten()
            
            # Find indices of zero and non-zero values
            zero_indices = torch.nonzero(im_f == 0.0).squeeze()
            nonzero_indices = torch.nonzero(im_f != 0.0).squeeze()
            
            # Only proceed if there are both zero and non-zero values
            if zero_indices.numel() > 0 and nonzero_indices.numel() > 0:
                # Convert to numpy for interpolation
                zero_indices_np = zero_indices.cpu().numpy()
                nonzero_indices_np = nonzero_indices.cpu().numpy()
                nonzero_values_np = im_f[nonzero_indices].cpu().numpy()
                
                # Interpolate
                interpolated_values = torch.tensor(
                    np.interp(zero_indices_np, nonzero_indices_np, nonzero_values_np),
                    dtype=im.dtype,
                    device=im.device
                )
                
                # Replace zero values with interpolated values
                im_f[zero_indices] = interpolated_values
            
            # Handle NaN values similarly
            nan_indices = torch.nonzero(torch.isnan(im_f)).squeeze()
            notnan_indices = torch.nonzero(~torch.isnan(im_f)).squeeze()
            
            if nan_indices.numel() > 0 and notnan_indices.numel() > 0:
                nan_indices_np = nan_indices.cpu().numpy()
                notnan_indices_np = notnan_indices.cpu().numpy()
                notnan_values_np = im_f[notnan_indices].cpu().numpy()
                
                interpolated_values = torch.tensor(
                    np.interp(nan_indices_np, notnan_indices_np, notnan_values_np),
                    dtype=im.dtype,
                    device=im.device
                )
                
                im_f[nan_indices] = interpolated_values
            
            # Reshape back
            im_slice = im_f.reshape(im_slice.shape)
            im[b, c] = im_slice
    
    return im

def torch_median_filter(im, kernel_size):
    """
    Apply a median filter to a PyTorch tensor
    
    Parameters
    ----------
    im : torch.Tensor
        Input tensor with shape (batch, channel, height, width)
    kernel_size : int
        Size of the median filter kernel
    
    Returns
    -------
    torch.Tensor
        Filtered tensor
    """
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    padding = kernel_size // 2
    batch_size, channels, height, width = im.shape
    
    # Initialize output tensor
    output = torch.zeros_like(im)
    
    # Process each batch and channel
    for b in range(batch_size):
        for c in range(channels):
            # Convert to numpy, apply median filter, and convert back to torch
            # Using numpy's median_filter as PyTorch doesn't have a built-in equivalent
            im_np = im[b, c].cpu().numpy()
            from scipy.ndimage import median_filter as scipy_median_filter
            filtered_np = scipy_median_filter(im_np, size=kernel_size)
            output[b, c] = torch.tensor(filtered_np, dtype=im.dtype, device=im.device)
    
    return output

def torch_afterglow_correction(im):
    """Correct dead pixels by adaptive median filtering for PyTorch tensors
    
    Parameters
    ----------
    im : torch.Tensor
        Image data (sinogram) as torch tensor with shape (batch, channel, height, width)
    
    Returns
    -------
    im : torch.Tensor
        Corrected image data
    """
    # Create a copy to avoid modifying the input
    im = im.clone()
    
    # Get the machine epsilon for float32
    eps = torch.finfo(torch.float32).eps
    print(eps)
    
    # Process each batch and channel
    for b in range(im.shape[0]):
        for c in range(im.shape[1]):
            im_slice = im[b, c]
            
            # Apply adaptive median filtering
            size_ct = 3
            while (torch.min(im_slice).item() < 0.0) and (size_ct <= 7):
                im_f = torch_median_filter(im_slice.unsqueeze(0).unsqueeze(0), size_ct).squeeze()
                im_slice[im_slice < 0.0] = im_f[im_slice < 0.0]
                size_ct += 2
            
            # Replace remaining negative/small values with average value
            if torch.min(im_slice).item() < eps:
                valid_mask = im_slice > eps
                if valid_mask.sum() > 0:  # Ensure there are valid pixels
                    rplc_value = im_slice[valid_mask].sum() / valid_mask.sum()
                    im_slice[im_slice < eps] = rplc_value
            
            im[b, c] = im_slice
    
    return im

def torch_interp(x, xp, fp, left=None, right=None):
    """
    One-dimensional linear interpolation for PyTorch tensors
    
    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates at which to evaluate the interpolated values
    xp : torch.Tensor
        The x-coordinates of the data points, must be increasing
    fp : torch.Tensor
        The y-coordinates of the data points, same length as xp
    left : float or None
        Value to return for x < xp[0], default is fp[0]
    right : float or None
        Value to return for x > xp[-1], default is fp[-1]
    
    Returns
    -------
    torch.Tensor
        The interpolated values
    """
    # Convert to CPU numpy for interpolation (PyTorch doesn't have a built-in equivalent)
    x_np = x.cpu().numpy()
    xp_np = xp.cpu().numpy()
    fp_np = fp.cpu().numpy()
    
    # Perform interpolation
    result_np = np.interp(x_np, xp_np, fp_np, left, right)
    
    # Convert back to tensor on the original device
    return torch.tensor(result_np, dtype=x.dtype, device=x.device)

# Example usage demonstration
def example_usage():
    # Create a sample batch of images
    batch_size = 2
    channels = 1
    height = 64
    width = 64
    
    # Create sample data with some dead pixels
    im = torch.rand(batch_size, channels, height, width, dtype=torch.float32)
    
    # Add some negative values and NaNs to simulate dead pixels
    dead_mask = torch.rand(batch_size, channels, height, width) < 0.05
    im[dead_mask] = -0.1
    
    nan_mask = torch.rand(batch_size, channels, height, width) < 0.02
    im[nan_mask] = torch.tensor(float('nan'))
    
    # Apply corrections
    corrected_dead = torch_dead_correction(im)
    corrected_afterglow = torch_afterglow_correction(im)
    
    return corrected_dead, corrected_afterglow

def tiehom_plan(im, beta, delta, energy, distance, pixsize, padding):
    """Pre-compute data to save time in further execution of phase_retrieval with TIE-HOM 
    (Paganin's) algorithm. Adjusted for PyTorch with batch processing.
    
    Parameters 
    ---------- 
    im : torch.Tensor
        Image tensor with shape (batch, channel, height, width). Channel should be 1.
    
    beta : float
        Imaginary part of the complex X-ray refraction index. 
    
    delta : float
        Decrement from unity of the real part of the complex X-ray refraction index. 
    
    energy [KeV]: float
        Energy in KeV of the incident X-ray beam. 
    
    distance [mm]: float
        Sample-to-detector distance in mm. 
    
    pixsize [mm]: float
        Size in mm of the detector element. 
    
    padding : bool
        Apply image padding to better process the boundary of the image
    """
    # Get additional values:
    lam = (12.398424 * 10 ** (-7)) / energy  # in mm
    mu = 4 * math.pi * beta / lam
    
    # Get dimensions from tensor
    _, _, dim0_o, dim1_o = im.shape
    
    # Replicate pad image if required:
    if padding:
        n_pad0 = dim0_o + dim0_o // 2
        n_pad1 = dim1_o + dim1_o // 2
    else:
        n_pad0 = dim0_o
        n_pad1 = dim1_o
    
    # Ensure even size:
    if n_pad0 % 2 == 1:
        n_pad0 = n_pad0 + 1
    if n_pad1 % 2 == 1:
        n_pad1 = n_pad1 + 1
    
    # Set the transformed frequencies according to pixelsize:
    rows = n_pad0
    cols = n_pad1
    
    ulim = torch.arange(-(cols) / 2, (cols) / 2, device=im.device)
    ulim = ulim * (2 * math.pi / (cols * pixsize))
    
    vlim = torch.arange(-(rows) / 2, (rows) / 2, device=im.device)
    vlim = vlim * (2 * math.pi / (rows * pixsize))
    
    v, u = torch.meshgrid(vlim, ulim, indexing='ij')
    
    # Apply formula:
    # Avoid division by zero with small epsilon
    den = 1 + distance * delta / mu * (u * u + v * v) + torch.finfo(torch.float32).eps
    
    # Shift the denominator (fftshift equivalent)
    den = torch.fft.fftshift(den)
    
    # For rfft, we only need half frequencies plus one
    den = den[:, :den.shape[1]//2 + 1]
    
    return {
        'dim0': dim0_o, 
        'dim1': dim1_o, 
        'npad0': n_pad0, 
        'npad1': n_pad1, 
        'den': den, 
        'mu': mu
    }


def pad_image(im, n_pad0, n_pad1):
    """Pad image to specified dimensions with pytorch.
    
    Parameters
    ----------
    im : torch.Tensor
        Image tensor with shape (batch, channel, height, width)
    n_pad0 : int
        Target height
    n_pad1 : int
        Target width
    """
    batch_size, channels, height, width = im.shape
    
    pad_height = n_pad0 - height
    pad_width = n_pad1 - width
    
    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # PyTorch padding is (left, right, top, bottom)
    padded = F.pad(im, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded


def tiehom(im, plan):
    """Process tomographic projection images with the TIE-HOM (Paganin's) phase retrieval algorithm.
    Adapted for batch processing with PyTorch.
    
    Parameters
    ----------
    im : torch.Tensor
        Flat corrected image data as tensor with shape (batch, channel, height, width).
        Channel should be 1.
    
    plan : dict
        Dictionary with pre-computed data (see tiehom_plan function).
    """
    # Extract plan values:
    dim0_o = plan['dim0']
    dim1_o = plan['dim1']
    n_pad0 = plan['npad0']
    n_pad1 = plan['npad1']
    marg0 = (n_pad0 - dim0_o) // 2
    marg1 = (n_pad1 - dim1_o) // 2
    den = plan['den']
    mu = plan['mu']
    
    # Ensure we're working with float32
    im = im.to(torch.float32)
    
    # Pad image (if required):
    im = pad_image(im, n_pad0, n_pad1)
    
    # Batch FFT processing
    # Squeeze the channel dimension for the FFT
    batch_size = im.shape[0]
    im = im.squeeze(1)  # Now shape is (batch, height, width)
    
    # Apply FFT
    im = torch.fft.rfft2(im)
    
    # Expand den to match batch dimension
    den_expanded = den.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Apply Paganin's formula
    im = im / den_expanded
    
    # Apply inverse FFT
    im = torch.fft.irfft2(im)
    
    # Add channel dimension back
    im = im.unsqueeze(1)  # Now shape is (batch, 1, height, width)
    
    # Apply log and scale
    # Adding small epsilon to avoid log(0)
    epsilon = torch.finfo(torch.float32).eps
    im = -1 / mu * torch.log(im + epsilon)
    
    # Return cropped output:
    return im[:, :, marg0:dim0_o + marg0, marg1:dim1_o + marg1]

def flatfield_correction(images, img_dark_before, img_dark_after, img_flat_before, img_flat_after, 
                         epsilon=2e-6, batch_size=100, device="cuda"):
    """
    Applies flat-field correction with linear interpolation of dark and flat-field images.
    Corrects dead pixels and does afterglow and ring artifact removal using memory-efficient batch processing.

    :param images: Tensor of shape (nr_img, C, W, H) - The images to correct.
    :param img_dark_before: Tensor (C, W, H) - Dark-field before measurement.
    :param img_dark_after: Tensor (C, W, H) - Dark-field after measurement.
    :param img_flat_before: Tensor (C, W, H) - Flat-field before measurement.
    :param img_flat_after: Tensor (C, W, H) - Flat-field after measurement.
    :param epsilon: Epsilon in division term for numerical stability.
    :param batch_size: Number of images to process at once (reduces memory usage).
    :return: Corrected images of shape (nr_img, C, W, H).
    """
    nr_img, C, W, H = images.shape  # Get number of images
    print("start flatfielding")

    # Apply dead correction on static images before processing
    img_dark_before = dead_correction(img_dark_before.to(device))
    img_dark_after = dead_correction(img_dark_after.to(device))
    img_flat_before = dead_correction(img_flat_before.to(device))
    img_flat_after = dead_correction(img_flat_after.to(device))
    epsilon = torch.tensor(epsilon).to(device)

    # Allocate output tensor
    img_corrected = torch.zeros_like(images)

    # Process images in batches
    for i in tqdm(range(0, nr_img, batch_size)):
        # Get batch slice
        batch = images[i : i + batch_size].to(device)

        # Apply dead correction on batch
        batch = dead_correction(batch)

        # Compute interpolation weights for this batch
        weights = torch.linspace(i, i + batch.shape[0] - 1, steps=batch.shape[0], device=batch.device) / (nr_img - 1)
        weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting

        # Interpolate dark & flat images
        img_dark = img_dark_before * (1 - weights) + img_dark_after * weights
        img_flat = img_flat_before * (1 - weights) + img_flat_after * weights

        # Perform flat-field correction: img_cor = (img - img_dark) / (img_flat - img_dark) + epsilon
        batch_corrected = torch.add(torch.div(batch - img_dark, torch.add(img_flat - img_dark, epsilon)), epsilon)

        # Apply artifact corrections
        batch_corrected = afterglow_correction(batch_corrected)  # Afterglow correction

        # Store results
        img_corrected[i : i + batch.shape[0]] = batch_corrected.detach().cpu()
  

    return img_corrected  # (nr_img, C, W, H)

    
def median_filter_1d(signal: torch.Tensor, kernel_size: int):
    """
    Applies a 1D median filter along the last dimension (width) of a tensor.

    :param signal: Input tensor of shape (B, C, W).
    :param kernel_size: Kernel size for median filtering (must be odd).
    :return: Median-filtered tensor of shape (B, C, W) (same shape as input).
    """
    pad = kernel_size // 2  # Padding to maintain output size

    # Apply padding before unfolding
    signal_padded = F.pad(signal, (pad, pad), mode='reflect')  # Shape: (B, C, W + 2*pad)

    # Unfold to extract sliding windows
    unfolded = signal_padded.unfold(dimension=-1, size=kernel_size, step=1)  # Shape: (B, C, W, K)

    return unfolded.median(dim=-1).values  # Compute median along the window

def rivers(im: torch.Tensor, args):
    """
    Applies the Rivers de-striping algorithm on a batched sinogram tensor.

    :param im: Input tensor of shape (B, C, H, W).
    :param args: Integer specifying median filter size (not a string anymore).
    :return: Corrected tensor of shape (B, C, H, W).
    """
    # Ensure args is an integer (not a string)
    if isinstance(args, str):
        n = int(args.split(";")[0])
    else:
        n = int(args)  # If passed as integer, use directly

    # Compute mean of each column (mean along height dimension H)
    col_mean = im.mean(dim=2, keepdim=True)  # Shape: (B, C, 1, W)

    # Apply 1D median filtering along width (W), ensuring padding preserves shape
    flt_col = median_filter_1d(col_mean.squeeze(2), n).unsqueeze(2)  # Shape: (B, C, 1, W)

    # Apply compensation: Adjust each row by subtracting (col_mean - filtered_col)
    im = im - (col_mean - flt_col)

    return im


def median_blur(im: torch.Tensor, kernel_size: int = 3):
    """
    Applies a median filter to a batched image tensor.

    :param im: Input tensor of shape (B, C, H, W).
    :param kernel_size: Kernel size for median filtering (must be odd).
    :return: Median-filtered tensor of the same shape.
    """
    B, C, H, W = im.shape
    pad = kernel_size // 2  # Padding for same-size output
    
    # Unfold (extract sliding windows)
    im_unfold = F.unfold(im, kernel_size, padding=pad)  # Shape: (B, C*ks*ks, L)
    
    # Reshape to (B, C, ks*ks, H*W)
    im_unfold = im_unfold.view(B, C, kernel_size * kernel_size, H * W)
    
    # Compute median along the window dimension
    im_median = im_unfold.median(dim=2).values  # Shape: (B, C, H*W)
    
    # Reshape back to (B, C, H, W)
    return im_median.view(B, C, H, W)
    
def despeckle_batched_optimized(img, batch_size=4, th=0.0005, device='cuda'):
    """
    Memory-efficient and optimized version of despeckle function for large image batches.
    
    Parameters:
    -----------
    img : torch.Tensor
        Input tensor with shape (batch, channel, height, width)
    batch_size : int
        Size of sub-batches to process at once
    th : float
        Threshold for identifying bad pixels
        
    Returns:
    --------
    torch.Tensor
        Despeckled image with same shape as input
    """
    # Get original device and shape
    original_batch_size = img.shape[0]
    
    # Pre-compute the Gaussian kernel once (shared for all batches)
    kernel_size = max(3, img.shape[2] // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gaussian_kernel = create_gaussian_kernel(kernel_size, img.shape[1], device)
    
    # Create result tensor with same shape and dtype as input
    result = torch.zeros_like(img)
    
    # Process in smaller batches
    for i in tqdm(range(0, original_batch_size, batch_size), desc="Despeckling batches"):
        # Get current batch indices
        batch_start = i
        batch_end = min(i + batch_size, original_batch_size)
        
        # Process current batch
        current_batch = img[batch_start:batch_end]
        processed_batch = despeckle_optimized(current_batch.to(device), gaussian_kernel, th)
        
        # Store result
        result[batch_start:batch_end] = processed_batch
        
        # Free memory explicitly
        torch.cuda.empty_cache()
    
    return result

def create_gaussian_kernel(kernel_size, channels, device):
    """Create a Gaussian kernel for the specified parameters"""
    sigma = kernel_size / 6.0
    
    # Create 1D Gaussian kernel (much faster than 2D grid approach)
    ax = torch.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    
    # Normalize the kernel
    kernel = kernel / kernel.sum()
    
    # Reshape to depthwise convolutional weight
    gaussian_kernel = kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel

def despeckle_optimized(img, gaussian_kernel, th=0.0005):
    """
    Optimized version of despeckle function.
    
    Parameters:
    -----------
    img : torch.Tensor
        Input tensor with shape (batch, channel, height, width)
    gaussian_kernel : torch.Tensor
        Pre-computed Gaussian kernel
    th : float
        Threshold for identifying bad pixels
        
    Returns:
    --------
    torch.Tensor
        Despeckled image with same shape as input
    """
    # Apply Gaussian filter
    kernel_size = gaussian_kernel.shape[2]
    padding = kernel_size // 2
    img_gauss = F.conv2d(
        img,
        gaussian_kernel,
        padding=padding,
        groups=img.shape[1]  # Depthwise convolution
    )
    
    # Find pixels that differ significantly from their neighborhood
    bad_pixels = torch.abs(img - img_gauss) > th
    
    # If no bad pixels, return the original image immediately
    if not bad_pixels.any():
        return img
    
    # Create result tensor - only create a copy if we need to modify it
    result = img.clone()
    
    # Create mask for each batch/channel that has bad pixels to avoid unnecessary computation
    batch_size, channels = img.shape[0], img.shape[1]
    has_bad_pixels = bad_pixels.view(batch_size, channels, -1).any(dim=2)
    
    # Use torch.median directly on patches for faster median filtering
    # We'll only process channels that actually have bad pixels
    for b in range(batch_size):
        for c in range(channels):
            if has_bad_pixels[b, c]:
                # Only process this channel if it contains bad pixels
                # Use F.unfold to extract patches efficiently
                img_pad = F.pad(img[b:b+1, c:c+1], pad=(1, 1, 1, 1), mode='reflect')
                patches = F.unfold(img_pad, kernel_size=3, stride=1)
                # Reshape to [batch, 9, H*W]
                patches = patches.reshape(1, 9, -1)
                # Get median of each 3x3 patch
                median_values = torch.median(patches, dim=1)[0]
                # Reshape back to image dimensions
                median_values = median_values.reshape(1, 1, img.shape[2], img.shape[3])
                # Only replace the bad pixels in this channel
                result[b:b+1, c:c+1][bad_pixels[b:b+1, c:c+1]] = median_values[bad_pixels[b:b+1, c:c+1]]
    
    return result    
def despeckle_batched(img, batch_size=4, th=0.0005, device='cuda'):
    """
    Find dead/bad pixels in batched images using PyTorch, processing in smaller batches
    to avoid CUDA out-of-memory errors.
    
    Parameters:
    -----------
    img : torch.Tensor
        Input tensor with shape (batch, channel, height, width)
    batch_size : int
        Size of sub-batches to process at once
    th : float
        Threshold for identifying bad pixels
        
    Returns:
    --------
    torch.Tensor
        Despeckled image with same shape as input
    """
    # Get original device and shape
    original_batch_size = img.shape[0]
    
    # Create result tensor with same shape and dtype as input
    result = torch.zeros_like(img)
    
    # Process in smaller batches
    for i in tqdm(range(0, original_batch_size, batch_size), desc="Despeckling batches"):
        # Get current batch indices
        batch_start = i
        batch_end = min(i + batch_size, original_batch_size)
        
        # Process current batch
        current_batch = img[batch_start:batch_end]
        processed_batch = despeckle(current_batch.to(device), th)
        
        # Store result
        result[batch_start:batch_end] = processed_batch
        
        # Optional: free memory explicitly
        torch.cuda.empty_cache()
    
    return result

def despeckle(img, th=0.0005):
    """
    Find dead/bad pixels in batched images using PyTorch
    
    Parameters:
    -----------
    img : torch.Tensor
        Input tensor with shape (batch, channel, height, width)
    th : float
        Threshold for identifying bad pixels
        
    Returns:
    --------
    torch.Tensor
        Despeckled image with same shape as input
    """
    # Make sure input is on the correct device
    device = img.device
    
    # Create Gaussian kernel for smoothing
    kernel_size = max(3, img.shape[2] // 10)
    # Make kernel_size odd if it's even
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    sigma = kernel_size / 6.0  # Approximation to get similar results to nd.gaussian_filter
    
    # Create gaussian kernel
    x_cord = torch.arange(kernel_size, device=device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    mean = (kernel_size - 1) / 2.
    variance = sigma**2
    
    # Calculate the 2-dimensional gaussian kernel
    gaussian_kernel = torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(img.shape[1], 1, 1, 1)
    
    # Apply Gaussian filter to each channel separately
    padding = kernel_size // 2
    img_gauss = F.conv2d(
        img,
        gaussian_kernel,
        padding=padding,
        groups=img.shape[1]  # Depthwise convolution
    )
    
    # Memory optimization: calculate difference and identify bad pixels before median filter
    bad_pixels = torch.abs(img - img_gauss) > th
    
    # Only calculate median filter if needed (if any bad pixels were found)
    if bad_pixels.any():
        # For median filter, use a sliding window approach with a smaller kernel
        median_kernel_size = 3  # Using a 3x3 window for median
        img_med = F.pad(img, [median_kernel_size//2]*4, mode='reflect')
        
        batch_size, channels, height, width = img.shape
        img_med_result = torch.zeros_like(img)
        
        # Process each batch and channel separately for median filter to save memory
        for b in range(batch_size):
            for c in range(channels):
                # Only process this batch/channel if it contains bad pixels
                if bad_pixels[b, c].any():
                    # Unfold only this channel to extract patches
                    channel_img = img_med[b:b+1, c:c+1]
                    patches = F.unfold(channel_img, kernel_size=median_kernel_size, padding=0, stride=1)
                    
                    # Reshape to get patches for each pixel
                    patches = patches.view(1, 1, median_kernel_size*median_kernel_size, height, width)
                    
                    # Get median of each patch
                    img_med_result[b, c] = patches.median(dim=2)[0]
        
        # Create result tensor
        result = img.clone()
        
        # Replace bad pixels with median filtered values
        result[bad_pixels] = img_med_result[bad_pixels]
        
        return result
    else:
        # If no bad pixels, return the original image
        return img
def afterglow_correction(im: torch.Tensor):
    """
    Corrects dead pixels in a batched multi-channel image tensor using adaptive median filtering.

    :param im: Input tensor of shape (B, C, H, W).
    :return: Corrected tensor of the same shape.
    """
    B, C, H, W = im.shape
    device = im.device  # Keep on the same device

    # Adaptive median filtering for isolated spots
    size_ct = 3
    while (torch.amin(im) < 0.0) and (size_ct <= 7):
        im_filtered = median_blur(im, size_ct)  # Apply custom median filtering
        im = torch.where(im < 0.0, im_filtered, im)  # Replace negative values
        size_ct += 2  # Increase kernel size for next iteration

    # Replace remaining negative values with the mean of positive values
    min_val = torch.amin(im)
    if min_val < torch.finfo(torch.float32).eps:  
        # Compute mean of positive values
        positive_mask = im > torch.finfo(torch.float32).eps
        mean_value = im[positive_mask].mean()
        
        # Replace negative values with computed mean
        im = torch.where(im < torch.finfo(torch.float32).eps, mean_value, im)

    return im


def dead_correction(im: torch.Tensor):
    """
    Corrects dead pixels (zeros & NaNs) in a batched multi-channel image tensor using interpolation.

    :param im: Input tensor of shape (B, C, H, W).
    :return: Corrected tensor of the same shape.
    """
    B, C, H, W = im.shape
    device = im.device  # Keep on the same device

    # Ensure no negative values (set negatives to zero)
    im = torch.maximum(im, torch.tensor(0.0, device=device))

    # Find dead pixels (zeros or NaNs)
    mask = (im == 0) | torch.isnan(im)  # Shape: (B, C, H, W)

    if mask.any():  # Only process if there are dead pixels
        # Replace NaNs with zero temporarily to prevent issues
        print('found faulty pixels')
        im = torch.nan_to_num(im, nan=0.0)

        # Create interpolation mask (1 for valid, 0 for missing pixels)
        interp_mask = (~mask).float()  # Shape: (B, C, H, W)

        # Use bilinear interpolation to estimate missing values
        im_interp = F.interpolate(im, scale_factor=1.0, mode='bilinear', align_corners=True)
        mask_interp = F.interpolate(interp_mask, scale_factor=1.0, mode='bilinear', align_corners=True)

        # Avoid division by zero (where all pixels are missing)
        mask_interp = torch.where(mask_interp == 0, torch.tensor(1.0, device=device), mask_interp)

        # Compute final corrected values
        im_corrected = im * interp_mask + im_interp * (1 - interp_mask)

        return im_corrected

    return im  # Return original if no correction was needed


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

    # Compute cropping start points (center crop)
    start_x = (W - crop_w) // 2
    start_y = (H - crop_h) // 2

    # Crop the image (batched indexing)
    cropped_image = image[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]  # (B, C, crop_h, crop_w)

    # Create a circular mask (only need to compute it once per crop size)
    device = image.device  # Ensure mask is on the same device
    y, x = torch.meshgrid(
        torch.arange(crop_h, device=device), torch.arange(crop_w, device=device), indexing="ij"
    )
    center_y, center_x = crop_h // 2, crop_w // 2
    radius = min(center_y, center_x)  # Radius to fit within cropped area

    mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2).float()  # (crop_h, crop_w)

    # Expand mask to match batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, crop_h, crop_w)
    mask = mask.expand(B, C, -1, -1)  # (B, C, crop_h, crop_w)

    # Apply the mask
    masked_image = cropped_image * mask  # Element-wise multiplication

    return masked_image  # (B, C, crop_h, crop_w)

def correct_center_shift(sinogram: torch.Tensor, shift):
    """
    Corrects center shift in a batched sinogram using the Fourier shift theorem.
    
    :param sinogram: Tensor of shape (B, C, num_angles, num_detectors).
    :param shift: Single float or tensor of shape (B, C).
    :return: Shifted sinogram of the same shape.
    """
    B, C, A, D = sinogram.shape  # Get batch, channels, angles, detectors

    # Convert shift to a tensor and ensure correct shape
    if isinstance(shift, (int, float)):  # Single scalar shift
        shift = torch.full((B, C, 1, 1), shift, device=sinogram.device, dtype=torch.float32)
    else:  # Tensor input (B, C)
        shift = shift.to(sinogram.device).view(B, C, 1, 1)  # Expand to (B, C, 1, 1)

    # Create frequency axis (1D FFT frequency values)
    freq = torch.fft.fftfreq(D, d=1, device=sinogram.device)  # (D,)
    freq = freq.view(1, 1, 1, D)  # Reshape for broadcasting

    # Compute phase shift in Fourier space
    phase_shift = torch.exp(-2j * torch.pi * shift * freq)  # (B, C, 1, D)

    # Apply FFT along the detector axis
    sinogram_fft = torch.fft.fft(sinogram, dim=-1)  # (B, C, A, D)

    # Apply the phase shift
    shifted_sinogram_fft = sinogram_fft * phase_shift  # Element-wise multiplication

    # Inverse FFT to get back to real space
    shifted_sinogram = torch.fft.ifft(shifted_sinogram_fft, dim=-1).real  # Take only the real part

    return shifted_sinogram  # (B, C, A, D)


