import torch
import torch.nn.functional as F

def find_best_overlap(img_left, img_right, min_overlap=400, max_overlap=500):
    """
    Finds the best overlap width by minimizing the Mean Squared Error (MSE).

    :param img_left: Tensor of shape (B, C, H, W1)
    :param img_right: Tensor of shape (B, C, H, W2)
    :param max_overlap: Maximum overlap width to consider.
    :return: Best overlap width.
    """
    B, C, H, W1 = img_left.shape
    W2 = img_right.shape[-1]

    # Limit max_overlap to avoid exceeding image width
    max_overlap = min(max_overlap, W1, W2)

    errors = []

    for overlap in range(min_overlap + 1, max_overlap + 1):
        # Compare the overlapping regions
        region1 = img_left[..., -overlap:]  # Last `overlap` columns of img1
        region2 = img_right[..., :overlap]   # First `overlap` columns of img2

        # Compute Mean Squared Error (MSE)
        mse = F.mse_loss(region1, region2)
        errors.append(mse)

    # Find the overlap width with the lowest MSE
    best_overlap = torch.argmin(torch.tensor(errors)).item() +  min_overlap + 1 # since overlap starts from min_overlap + 1
    return best_overlap

def stitch_images(img1, img2, overlap=100):
    """
    Takes given overlap and stitches two images horizontally.

    :param img1: Tensor of shape (B, C, H, W1)
    :param img2: Tensor of shape (B, C, H, W2)
    :param overlap: Overlap of the two images.
    :return: Stitched image tensor of shape (B, C, H, W1 + W2 - best_overlap).
    """

    # Blend transition using linear alpha blending
    alpha = torch.linspace(0, 1, overlap, device=img1.device).view(1, 1, 1, -1)
    blend_region = img1[..., -overlap:] * (1 - alpha) + img2[..., :overlap] * alpha

    # Concatenate the blended and non-overlapping parts
    stitched = torch.cat([img1[..., :-overlap], blend_region, img2[..., overlap:]], dim=-1)
    return stitched

# Example Usage
# B, C, H, W1, W2 = 2, 3, 128, 300, 320  # Example sizes
# img1 = torch.randn(B, C, H, W1)
# img2 = torch.randn(B, C, H, W2)
# stitched_img = stitch_images(img1, img2, max_overlap=50)

