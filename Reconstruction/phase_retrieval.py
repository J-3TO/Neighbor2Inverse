import torch
import numpy as np
import random
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def paganin_kernel(shape: tuple, sigma: float,
                           pixel_size: float = 0.172) -> torch.Tensor:
    """
    Compute kernel for Paganins phase retrieval.

    Args:
        shape (tuple): projection shape (..., height, width)
        sigma (float): delta / mu * z
        pixel_size (float, optional): sampling period. Defaults to 0.172.

    Returns:
        torch.Tensor: kernel
    """

    u = torch.fft.fftfreq(shape[-1], d=pixel_size).to(device)
    v = torch.fft.fftfreq(shape[-2], d=pixel_size).to(device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    return 1 / (sigma * 4 * torch.pi**2 * (uu**2 + vv**2) + 1)


def gaussian_kernel(shape: tuple, sigma: float) -> torch.Tensor:
    """_summary_

    Args:
        shape (tuple): projection shape (height, width)
        sigma (float): standard deviation

    Returns:
        torch.Tensor: kernel
    """
    u = torch.fft.fftfreq(shape[-1]).to(device)
    v = torch.fft.fftfreq(shape[-2]).to(device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    return torch.exp(-4 * np.pi**2 * (uu**2 + vv**2) * sigma**2 * 0.5)


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
    return torch.fft.ifft2(torch.einsum(
        "hijk, jk -> hijk", torch.fft.fft2(proj), kernel)).real


# params: sigma = linspace(0.05, 4)
def compute_phase_retrieval(proj: torch.Tensor, sigma: float,
                            pixel_size: float=0.172) -> torch.Tensor:
    """
    Compute Paganin phase retrieval.

    Args:
        proj (torch.Tensor): projections (..., w, h)
        sigma (float): simplified delta / mu * propagation_distance 
        pixel_size (float): pixel size. Defaults to 0.172.

    Returns:
        torch.Tensor: phase retrieved transmission images.
    """
    kernel = paganin_kernel(proj.shape, sigma, pixel_size).to(device)
    return convolve(proj, kernel)


# params: sigma = range(1, 3), iters = range(5, 11)
def compute_rl_deconvolution(proj: torch.Tensor, sigma: float, iters: int) -> torch.Tensor:
    """
    Compute Richardson-Lucy deconvolution.

    Args:
        proj (torch.Tensor): projections (..., w, h)
        sigma (float): standard deviation
        iters (int): iteration

    Returns:
        torch.Tensor: deconvolved projections
    """
    
    shape = (proj.shape[-2], proj.shape[-1])
    kernel = gaussian_kernel(shape, sigma).to(device)

    epsilon = 1e-8
    rl_proj = proj.clone()

    for i in range(iters):
        model = rl_proj.clone()
        model = convolve(model, kernel)
        model = proj / (model + epsilon)
        model = convolve(model, kernel)
        rl_proj *= model
    return rl_proj

# npad = 50? 
def padding(tensor: torch.Tensor, npad: int) -> np.ndarray:
    """
    Pad torch tensor to be periodic with cosine fade.

    Args:
        arr (torch.Tensor): tensor
        npad (int): pad size (in each dimension)

    Returns:
        np.ndarray: padded array
    """
    
    weight_major = 0.5 + 0.5 * torch.cos(
        torch.linspace(0., torch.pi * 0.5, npad)).to(device)
    weight_minor = 0.5 + 0.5 * torch.cos(
        torch.linspace(torch.pi, torch.pi * 0.5, npad)).to(device)

    ten_pad = torch.nn.functional.pad(tensor, (npad, npad, npad, npad))

    ten_pad[..., :npad] = \
        torch.flip(weight_major, (0,))[None, None, :]\
        * ten_pad[..., npad + 1][..., None]\
        + torch.flip(weight_minor, (0,))[None, None, :]\
        * ten_pad[..., -(npad + 1)][..., None]

    ten_pad[..., -npad:] = \
        weight_major[None, None, :]\
        * ten_pad[..., -(npad + 1)][..., None]\
        + weight_minor[None, None, :]\
        * ten_pad[..., npad + 1][..., None]

    ten_pad[:, :npad] = \
        torch.flip(weight_major, (0,))[None, :, None]\
        * ten_pad[:, npad + 1][:, None]\
        + torch.flip(weight_minor, (0,))[None, :, None]\
        * ten_pad[:, -(npad + 1)][:, None]

    ten_pad[:, -npad:] = \
        weight_major[None, :, None]\
        * ten_pad[:, -(npad + 1)][:, None]\
        + weight_minor[None, :, None]\
        * ten_pad[:, npad + 1][:, None]

    return ten_pad


@torch.no_grad()
def compute_paganin_batch(proj_stack,
                          delta=0.8e-8,       # Refractive index decrement
                          beta=1e-11,         # Absorption index
                          z=5000,             # Propagation distance (mm)
                          pixel_size=0.009,    # Detector pixel size (mm)
                          energy=70,          # X-ray energy (keV)
                          batch_size=8,       # Number of projections to process at once
                          device='cuda', 
                          clip = (None, None)):     # Computation device
    """
    Computes Paganin phase retrieval on a stack of projection images using a batched approach.
    
    This implementation uses the single-material phase retrieval algorithm developed by
    Paganin et al. for near-field phase contrast imaging. The algorithm recovers the phase
    information from intensity measurements taken at a single distance.
    
    Args:
        proj_stack (torch.Tensor): Stack of projection images [num_projections, height, width]
        delta (float): Refractive index decrement of the material
        beta (float): Absorption index of the material
        z (float): Propagation distance (mm)
        pixel_size (float): Detector pixel size (mm)
        energy (float): X-ray energy (keV)
        batch_size (int): Number of projections to process at once
        device (str): Computation device ('cuda' or 'cpu')
        clip (bool): Clip thickness value to specified values
        
    Returns:
        torch.Tensor: Stack of phase-retrieved images [num_projections, channel, height, width]
    
    Notes:
        - The function applies padding to handle edge effects in the Fourier transform
        - Small constant (2e-6) is added to avoid log of zero in the final calculation
        - Input projections are clamped to [0, 1] range before processing
    """
    
    # Calculate pad size
    pad_size = proj_stack.shape[-1] // 2
    
    # Calculate physical parameters
    wavelength = (12.398424 * 10**(-7)) / energy  # X-ray wavelength (mm)
    mu = 4 * np.pi * beta / wavelength            # Linear absorption coefficient
    sigma = delta / mu * z   
    print("sigma:", sigma, "delta/beta ratio:", delta/beta)
    
    
    phas_img_list = []
    # Calculate number of batches with ceiling division
    num_batches = (len(proj_stack) + batch_size - 1) // batch_size
    
    for k in tqdm(range(num_batches)):
        # Extract and prepare current batch
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, len(proj_stack))
        
        # Move batch to GPU, convert to float, and add small constant
        batch = proj_stack[start_idx:end_idx].to(device).float() + 2e-6
        
        # Apply padding to handle edge effects in Fourier transform
        # Ensure batch is in correct shape for padding function
        if len(batch.shape) == 3:  # [batch, height, width]
            batch_pad = padding(batch, pad_size)
        else:  # [batch, channels, height, width]
            batch_pad = padding(batch.view(batch.size(0), batch.size(2), batch.size(3)), pad_size)
        
        # Add channel dimension if needed
        if len(batch_pad.shape) == 3:
            batch_pad = batch_pad[:, None]
            
        # Apply phase retrieval algorithm
        batch_paganin = compute_phase_retrieval(batch_pad, sigma=sigma, pixel_size=pixel_size)
        
        # Remove padding
        batch_paganin = batch_paganin[:, :, pad_size:-pad_size, pad_size:-pad_size]

        if clip[0]!=None and clip[1]!=None:
            batch_paganin = torch.clamp(batch_paganin, min=clip[0], max=clip[1])
        
        # Calculate absorption projection from phase-retrieved intensity
        batch_paganin_log = -1 / mu * torch.log(batch_paganin)
        
        # Validate output (no NaN values)
        assert not torch.isnan(batch_paganin_log).any(), f"Found NaN at batch {k}, min values: batch={batch.min()}, phase={batch_paganin.min()}"
        
        # Store results
        phas_img_list.extend(batch_paganin_log.detach().cpu())
    
    # Stack all processed batches into a single tensor
    return torch.stack(phas_img_list, axis=0)
    
