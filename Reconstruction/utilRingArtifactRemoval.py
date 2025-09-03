import torch
from typing import Tuple, Optional
from tqdm import tqdm
import torch.nn.functional as F

def torch_sort_forward(tensor: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts voxel values of a tensor along a specified dimension (Height or Width).

    Operates on batches of tensors (B, H, W).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (B, H, W).
    dim : int
        Dimension along which to sort.
        Use 1 to sort along Height (H).
        Use 2 to sort along Width (W).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - sorted_tensor : torch.Tensor
            Tensor with values sorted along the specified dimension. Shape (B, H, W).
        - sort_indices : torch.Tensor
            Indices that store the original positions before sorting.
            These indices can be used with torch_sort_backward. Shape (B, H, W).
            These are the indices needed to *gather* from the original tensor
            to produce the sorted tensor.
    """
    if dim not in [1, 2]:
        raise ValueError("Dimension 'dim' must be 1 (Height) or 2 (Width)")

    # torch.sort returns sorted values and the indices required to reconstruct
    # the sort from the original tensor.
    sorted_tensor, indices_for_sort = torch.sort(tensor, dim=dim)

    # To get the indices needed for torch_sort_backward (i.e., the original
    # positions *corresponding* to each element in the sorted_tensor), we
    # create a grid of original indices and gather them using indices_for_sort.

    # Create a tensor representing the original indices along the sorting dimension
    b, h, w = tensor.shape
    if dim == 1: # Sort along Height
        # Indices range from 0 to H-1
        original_indices_shape = (1, h, 1)
        original_indices_arange = torch.arange(h, device=tensor.device, dtype=indices_for_sort.dtype).view(original_indices_shape)
        original_indices_expanded = original_indices_arange.expand(b, h, w) # Expand to (B, H, W)
    else: # Sort along Width (dim == 2)
        # Indices range from 0 to W-1
        original_indices_shape = (1, 1, w)
        original_indices_arange = torch.arange(w, device=tensor.device, dtype=indices_for_sort.dtype).view(original_indices_shape)
        original_indices_expanded = original_indices_arange.expand(b, h, w) # Expand to (B, H, W)

    # Gather the original indices based on how the values were sorted.
    # This tells us, for each position in sorted_tensor, where it came from originally.
    # This is the equivalent of `mat_index` in the numpy version.
    indices_backward = torch.gather(original_indices_expanded, dim=dim, index=indices_for_sort)

    return sorted_tensor, indices_backward


def torch_sort_backward(tensor: torch.Tensor, indices_backward: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sorts voxel values of a tensor back to their original positions using
    the index tensor provided by torch_sort_forward.

    Operates on batches of tensors (B, H, W).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (potentially modified, e.g., filtered) with shape (B, H, W).
        This tensor should have the same shape as the original tensor passed to
        torch_sort_forward, but its values might have changed. It is assumed
        to be in the 'sorted' state corresponding to indices_backward.
    indices_backward : torch.Tensor
        Index tensor obtained from torch_sort_forward. Shape (B, H, W).
        It indicates the original position (along 'dim') for each element
        in the input 'tensor'.
    dim : int
        Dimension along which the original sorting was performed.
        Use 1 for Height (H).
        Use 2 for Width (W).

    Returns
    -------
    torch.Tensor
        Tensor with values resorted to their original positions. Shape (B, H, W).
    """
    if dim not in [1, 2]:
        raise ValueError("Dimension 'dim' must be 1 (Height) or 2 (Width)")
    if tensor.shape != indices_backward.shape:
        raise ValueError("Input 'tensor' and 'indices_backward' must have the same shape.")

    # To unsort, we need indices that tell us, for each output position,
    # which element from the input 'tensor' (the sorted one) should go there.
    # This is achieved by argsorting the 'indices_backward' tensor.
    # The result 'unsort_indices' gives the indices to *gather* from the input
    # 'tensor' to restore the original order.
    unsort_indices = torch.argsort(indices_backward, dim=dim)

    # Gather elements from the input tensor using the unsort_indices
    unsorted_tensor = torch.gather(tensor, dim=dim, index=unsort_indices)

    return unsorted_tensor

def manual_median_filter_2d(input_tensor: torch.Tensor, kernel_size: tuple[int, int], padding_mode: str = 'reflect'):
    """
    Basic manual 2D median filter example using unfold.
    Assumes input (B, C, H, W). kernel_size (kH, kW) must be odd.
    """
    b, c, h, w = input_tensor.shape
    kh, kw = kernel_size
    assert kh % 2 == 1 and kw % 2 == 1, "Kernel dimensions must be odd"

    # Calculate padding size
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad the input tensor
    # Note: PyTorch padding format is (pad_left, pad_right, pad_top, pad_bottom)
    padded_input = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)

    # Unfold to extract patches
    # unfold(dimension, size, step)
    patches = padded_input.unfold(2, kh, 1).unfold(3, kw, 1)
    # Shape: (B, C, H_out, W_out, kH, kW) where H_out=H, W_out=W

    # Reshape to flatten the kernel dimensions
    patches = patches.contiguous().view(b, c, h, w, -1) # Shape: (B, C, H, W, kH*kW)

    # Compute the median along the patch dimension
    median_values, _ = torch.median(patches, dim=-1) # Shape: (B, C, H, W)

    return median_values

def sorted_filter_batch(
    sinograms: torch.Tensor,
    sort_dim: int,
    kernel_size: int,
    batch_size: int,
    device: str = 'cuda',
    padding_mode: str = 'reflect',
) -> torch.Tensor:
    
    if sinograms.ndim != 3:
        raise ValueError(f"Input sinograms tensor must have 3 dimensions (B, H, W), but got {sinograms.ndim}")
    if sort_dim not in [1, 2]:
        raise ValueError("Dimension 'sort_dim' must be 1 (Height) or 2 (Width)")
    if kernel_size % 2 == 0:
        raise ValueError(f"filter_kernel_size must be odd, but got {kernel_size}")

    out_list = []
    total_size = sinograms.shape[0]
    for i in tqdm(range(0, total_size, batch_size)):
        sinograms_batch = sinograms[i : i + batch_size]
        sino_sort, sino_indices = torch_sort_forward(sinograms_batch, dim=sort_dim)
        sino_sort_c = sino_sort.unsqueeze(1) # (B, 1, H, W)
    
        sino_sort_filtered_c = manual_median_filter_2d(sino_sort_c.cuda(), kernel_size=(1, kernel_size)).detach().cpu()
        sino_sort_filtered = sino_sort_filtered_c.squeeze(1) # (B, H, W)
        sino_filtered_unsorted = torch_sort_backward(sino_sort_filtered, sino_indices, dim=sort_dim).unsqueeze(1)
        out_list.extend(sino_filtered_unsorted)

    return torch.stack(out_list)