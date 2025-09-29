import torch
import torch.nn.functional as F
import scipy.constants as const # Use SciPy constants directly
import math
import numpy as np # Keep numpy for comparison if needed outside the function

# --- PyTorch compatible helper functions (using SciPy constants) ---

def _get_constants(device, dtype):
    """Gets physical constants as tensors on the correct device/dtype."""
    h = torch.tensor(const.h, dtype=dtype, device=device)
    eV = torch.tensor(const.eV, dtype=dtype, device=device)
    c = torch.tensor(const.c, dtype=dtype, device=device)
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    return h, eV, c, pi

def energy2wl_torch(energy_kev, device, dtype):
    """Convert energy [keV] to wavelength [m] using PyTorch."""
    h, eV, c, _ = _get_constants(device, dtype)
    energy_j = energy_kev * 1000.0 * eV
    # Add small epsilon to prevent division by zero if energy is zero
    wavelength = h * c / (energy_j + 1e-30)
    return wavelength

def wl2energy_torch(wavelength, device, dtype):
    """Convert photon wavelength [m] to energy [keV] using PyTorch."""
    h, eV, c, _ = _get_constants(device, dtype)
    # Add small epsilon to prevent division by zero if wavelength is zero
    energy_j = h * c / (wavelength + 1e-30)
    energy_kev = energy_j / eV / 1000.0
    return energy_kev

def energy2k_torch(energy_kev, device, dtype):
    """Convert energy [keV] to wavevector k [m^-1] using PyTorch."""
    _, _, _, pi = _get_constants(device, dtype)
    wavelength = energy2wl_torch(energy_kev, device, dtype)
    # Add small epsilon to prevent division by zero if wavelength is zero
    k = 2 * pi / (wavelength + 1e-30)
    return k

# --- Gradient Function (Mimicking np.gradient) ---

def gradient_torch(f, px):
    """
    Calculates the gradient of a 4D tensor (B, C, H, W) using finite differences,
    mimicking np.gradient behaviour (central differences, forward/backward at boundaries).
    Returns gradients along H (axis 2) and W (axis 3) scaled by pixel size.
    """
    B, C, H, W = f.shape
    # Use torch.gradient introduced in PyTorch 1.7+ (preferred)
    if hasattr(torch, 'gradient'):
        # torch.gradient computes central differences and forward/backward at boundary
        # It returns a tuple of tensors, one for each dimension specified.
        # We scale by pixel size here. Note: torch.gradient spacing defaults to 1.
        grad_y_unscaled, grad_x_unscaled = torch.gradient(f, dim=(2, 3))
        # Scale by pixel size (difference is over 2*px for central, 1*px for boundary)
        # torch.gradient handles the spacing correctly if specified, but we scale manually here for clarity like np.gradient output
        # However, np.gradient implicitly scales: dy = (f[i+1]-f[i-1]) / (2*dx)
        # torch.gradient(f, spacing=s) -> dy = (f[i+1]-f[i-1]) / (2*s)
        # So we just need to divide by px
        grad_y = grad_y_unscaled / px
        grad_x = grad_x_unscaled / px

    else:
        # Manual implementation for older PyTorch versions
        # Gradient along H (axis 2) - dy
        grad_y = torch.zeros_like(f)
        # Central difference
        grad_y[:, :, 1:-1, :] = (f[:, :, 2:, :] - f[:, :, :-2, :]) / (2 * px)
        # Forward difference at start boundary
        grad_y[:, :, 0, :] = (f[:, :, 1, :] - f[:, :, 0, :]) / px
        # Backward difference at end boundary
        grad_y[:, :, -1, :] = (f[:, :, -1, :] - f[:, :, -2, :]) / px

        # Gradient along W (axis 3) - dx
        grad_x = torch.zeros_like(f)
        # Central difference
        grad_x[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * px)
        # Forward difference at start boundary
        grad_x[:, :, :, 0] = (f[:, :, :, 1] - f[:, :, :, 0]) / px
        # Backward difference at end boundary
        grad_x[:, :, :, -1] = (f[:, :, :, -1] - f[:, :, :, -2]) / px

    return grad_y, grad_x # Return gradient along H, then W

# --- Laplacian Kernel (for Conv2d) ---

def _get_laplacian_kernel(dtype, device):
    """Gets a 3x3 Laplacian kernel for conv2d."""
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=dtype, device=device)
    # Reshape for conv2d: (out_channels=1, in_channels=1, H, W)
    return kernel.view(1, 1, 3, 3)


# --- Main Transformed Function ---

def propTIE_torch(
    delta,
    beta,
    energy,
    distance,
    px,
    batch_size,
    ind_terms=False,
    supersample=3,
    mode="TIE",
    use_float64=True # Added option for precision control
):
    """Forward propagation using the transport of intensity equation (PyTorch version).

    Handles inputs of shape (B, C, H, W) and processes in batches.
    Attempts to closely match the NumPy/SciPy version's numerics by using:
    - Reflect padding for Laplacian.
    - Finite differences mimicking np.gradient.
    - Optional float64 precision.

    Parameters
    ----------
    delta : torch.Tensor
        Projected delta distribution tensor of shape (B, C, H, W).
    beta : torch.Tensor
        Projected beta distribution tensor of shape (B, C, H, W).
    energy : float or torch.Tensor
        Energy in [keV]. If tensor, should be broadcastable to (B, C, 1, 1).
    distance : float
        Propagation distance in [m].
    px : float
        Pixelsize of field in [m].
    batch_size : int
        Number of images (from dimension B) to process in each batch.
    ind_terms : bool, optional
        If True the individual terms of the TIE are returned. Default is False.
    supersample : int, optional
        Upsampling factor for intermediate calculations. Default is 3.
        Highly recommended unless speed/memory is an issue. Set to 1 to disable.
    mode : str, optional
        TIE model to use ('TIE', 'TIE_nct', 'TIE_lin', 'TIE_lin_nct').
        Default is 'TIE'.
    use_float64 : bool, optional
        If True, performs calculations in float64 precision. Default is True.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Intensity tensor at the given distance, shape (B, C, H, W).
        If ind_terms is True, returns a tuple:
        (attenuation_term, laplacian_term, cross_term or None).
    """
    # --- Input Validation and Setup ---
    if not isinstance(delta, torch.Tensor) or not isinstance(beta, torch.Tensor):
        raise TypeError("delta and beta must be PyTorch Tensors")
    if delta.shape != beta.shape:
        raise ValueError("delta and beta must have the same shape")
    if delta.ndim != 4:
        raise ValueError("Input tensors delta and beta must have 4 dimensions (B, C, H, W)")

    # Set device and data type
    device = delta.device
    original_dtype = delta.dtype
    dtype = torch.float64 if use_float64 else original_dtype

    # Cast inputs if necessary
    delta = delta.to(dtype=dtype)
    beta = beta.to(dtype=dtype)

    B, C, H, W = delta.shape

    # Ensure energy is a tensor and broadcastable, on correct device/dtype
    if not isinstance(energy, torch.Tensor):
        energy = torch.tensor(energy, dtype=dtype, device=device)
    else:
        energy = energy.to(dtype=dtype, device=device)

    # Reshape energy for broadcasting: (B or 1, C or 1, 1, 1)
    if energy.ndim == 0:
        energy = energy.view(1, 1, 1, 1)
    elif energy.ndim == 1 and energy.shape[0] == B:
        energy = energy.view(B, 1, 1, 1)
    elif energy.ndim == 2 and energy.shape[0] == B and energy.shape[1] == C:
        energy = energy.view(B, C, 1, 1)
    # Ensure it can broadcast to (current_batch_size, C, 1, 1) later

    if distance == 0:
        supersample = 1 # No need to supersample if not propagating

    ss = supersample
    px_eff = px / ss # Effective pixel size after supersampling

    # Get Laplacian kernel, ensure it is on the correct device/dtype
    laplacian_kernel = _get_laplacian_kernel(dtype, device).repeat(C, 1, 1, 1)

    # Calculate wavevector k (can be different per batch/channel if energy varies)
    k = energy2k_torch(energy, device, dtype) # Shape (B or 1, C or 1, 1, 1)

    # --- Batch Processing ---
    results_list = []
    num_batches = math.ceil(B / batch_size)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, B)
        b_slice = slice(start_idx, end_idx) # Slice for the batch dimension
        current_batch_size = end_idx - start_idx

        # Slice input tensors and k for the current batch
        delta_batch = delta[b_slice]
        beta_batch = beta[b_slice]
        # Ensure k broadcasts correctly to the current batch size
        k_batch = k if k.shape[0] == 1 else k[b_slice]
        if k_batch.shape[0] != current_batch_size and k_batch.shape[0] != 1:
             raise ValueError("Energy/k tensor dimension mismatch with batch size")
        if k_batch.shape[1] != C and k_batch.shape[1] != 1:
             raise ValueError("Energy/k tensor channel mismatch")


        # --- Supersampling (Upsampling) ---
        if ss > 1:
            new_H, new_W = int(H * ss), int(W * ss)
            delta_batch_ss = F.interpolate(delta_batch, size=(new_H, new_W), mode='bicubic', align_corners=False)
            beta_batch_ss = F.interpolate(beta_batch, size=(new_H, new_W), mode='bicubic', align_corners=False)
        else:
            delta_batch_ss = delta_batch
            beta_batch_ss = beta_batch

        # --- Core TIE Calculations (on supersampled batch) ---
        # Phase term: phi = -k * delta
        phi = -k_batch * delta_batch_ss # k_batch broadcasts correctly

        # Laplacian of phase: Lphi = laplace(phi) / (px_eff^2)
        # Use reflect padding before convolution to match scipy.ndimage.laplace default
        pad_width = 1 # For 3x3 kernel
        phi_padded = F.pad(phi, (pad_width, pad_width, pad_width, pad_width), mode='reflect')
        # Convolve the padded tensor, NO padding in conv2d itself
        Lphi_unscaled = F.conv2d(phi_padded, laplacian_kernel, padding=0, groups=C)
        Lphi = Lphi_unscaled / (px_eff ** 2)

        # --- Mode-Specific Calculations ---
        attenuation_term_ss = None
        laplacian_term_ss = None
        cross_term_ss = None

        # Common gradient calculation (if needed) using finite differences
        # Note: We compute gradient on the *unscaled* phi or M or I0,
        # then scale by px_eff afterwards, matching the revised gradient_torch function
        if mode in ("TIE", "TIE_lin"):
             grad_phi_y, grad_phi_x = gradient_torch(phi, px=px_eff)

        if mode in ("TIE", "TIE_nct"):
            # Intensity without propagation: I0 = exp(-2 * k * beta)
            I0 = torch.exp(-2 * k_batch * beta_batch_ss)
            attenuation_term_ss = I0

            # Laplacian term: lap = distance / k * I0 * Lphi
            lap = (distance / k_batch) * I0 * Lphi
            laplacian_term_ss = lap

            if mode == "TIE":
                # Gradient of I0
                grad_I0_y, grad_I0_x = gradient_torch(I0, px=px_eff)

                # Cross term: ct = distance / k * (gradI0 . gradphi)
                ct = (distance / k_batch) * (grad_I0_x * grad_phi_x + grad_I0_y * grad_phi_y)
                cross_term_ss = ct
                result_batch_ss = I0 - lap - ct
            else: # TIE_nct
                result_batch_ss = I0 - lap
                # cross_term_ss remains None

        elif mode in ("TIE_lin", "TIE_lin_nct"):
            # Attenuation factor M = -2 * k * beta
            M = -2 * k_batch * beta_batch_ss
            attenuation_term_ss = M # Return M for linearized case if ind_terms=True

            # Laplacian term (linearized): lap = distance / k * Lphi
            lap = (distance / k_batch) * Lphi
            laplacian_term_ss = lap

            if mode == "TIE_lin":
                # Gradient of M
                grad_M_y, grad_M_x = gradient_torch(M, px=px_eff)

                # Cross term (linearized): ct = distance / k * (gradM . gradphi)
                ct = (distance / k_batch) * (grad_M_x * grad_phi_x + grad_M_y * grad_phi_y)
                cross_term_ss = ct
                result_batch_ss = torch.exp(M - lap - ct)
            else: # TIE_lin_nct
                result_batch_ss = torch.exp(M - lap)
                # cross_term_ss remains None

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # --- Downsampling (Zoom out) ---
        if ss > 1:
            # Interpolate back to original H, W
            result_batch = F.interpolate(result_batch_ss, size=(H, W), mode='bicubic', align_corners=False)
            if ind_terms:
                attenuation_term = F.interpolate(attenuation_term_ss, size=(H, W), mode='bicubic', align_corners=False)
                laplacian_term = F.interpolate(laplacian_term_ss, size=(H, W), mode='bicubic', align_corners=False)
                if cross_term_ss is not None:
                    cross_term = F.interpolate(cross_term_ss, size=(H, W), mode='bicubic', align_corners=False)
                else:
                    cross_term = None # Maintain None if no cross term
        else:
            result_batch = result_batch_ss
            if ind_terms:
                 attenuation_term = attenuation_term_ss
                 laplacian_term = laplacian_term_ss
                 cross_term = cross_term_ss # Can be None

        # --- Store results ---
        if ind_terms:
            # Ensure terms are cast back to original dtype before storing if needed
            results_list.append((attenuation_term.to(original_dtype),
                                 laplacian_term.to(original_dtype),
                                 cross_term.to(original_dtype) if cross_term is not None else None))
        else:
            # Cast result back to original dtype
            results_list.append(result_batch.to(original_dtype))

    # --- Concatenate results from all batches ---
    if ind_terms:
        # Concatenate each term separately
        attenuation_final = torch.cat([res[0] for res in results_list], dim=0)
        laplacian_final = torch.cat([res[1] for res in results_list], dim=0)
        # Handle potential None for cross_term
        if results_list[0][2] is not None:
            cross_term_final = torch.cat([res[2] for res in results_list], dim=0)
        else:
            cross_term_final = None
        return attenuation_final, laplacian_final, cross_term_final
    else:
        # Concatenate the final intensity result
        intensity_final = torch.cat(results_list, dim=0)
        return intensity_final