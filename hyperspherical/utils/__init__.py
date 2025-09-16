import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "conformal_point",
    "conformal_sphere",
    "conformal_product",
    "spheres_radii",
    "spheres_squared_radii",
    "spheres_centers",
    "conformal_filters",
    "conformal_einf_spheres",
    "compute_einf_tensor"
]


def conformal_point(x: torch.Tensor) -> torch.Tensor:
    """
    Conformalize a tensor by appending the squared norm of the tensor as an additional dimension.
    :param x: Tensor of shape (..., n), where n is the dimension of the input.
    :return: Tensor of shape (..., n + 1)
    """
    einf = 0.5 * torch.sum(torch.square(x), dim=-1, keepdim=True)
    return torch.cat((x, einf), dim=-1)


def conformal_sphere(center: torch.Tensor, radius: float | torch.FloatTensor) -> torch.Tensor:
    """
    Create a conformal sphere representation from its center and radius.
    :param center: Tensor of shape (..., n) representing the center of the sphere.
    :param radius: FloatTensor of shape (...) representing the radius of the sphere.
    :return: Tensor of shape (..., n + 1) representing the conformal sphere.
    """
    if isinstance(radius, float):
        radius = torch.tensor(radius, dtype=center.dtype, device=center.device)
    p_squared = torch.sum(torch.square(center), dim=-1)
    einf = 0.5 * (p_squared - torch.square(radius))
    return torch.cat((center, einf.unsqueeze(-1)), dim=-1)


def conformal_center(spheres: torch.Tensor) -> torch.Tensor:
    """
    Extract the center of a conformal sphere.
    :param spheres: Tensor of shape (..., n + 1) representing the conformal spheres.
    :return: Tensor of shape (..., n+1) representing the centers of the spheres.
    """
    return conformal_point(spheres[..., :-1])


def conformal_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the conformal product of two tensors.
    :param x: Tensor of shape (..., n + 1), where n is the dimension of the input.
    :param y: Tensor of shape (..., n + 1).
    :return: Tensor of shape (..., n) representing the conformal product.
    """
    ps = torch.einsum("...si,...bi->...bs", x[..., :-1], y[..., :-1])
    return ps - x[..., -1].view(1, -1) - y[..., [-1]]


def spheres_radii(spheres: torch.Tensor) -> torch.Tensor:
    """
    Compute the radius of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (...) representing the radius of the sphere.
    """
    return torch.sqrt(spheres_squared_radii(spheres).clamp(min=0.0))


def spheres_squared_radii(spheres: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared radius of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (...) representing the radius of the sphere.
    """
    norm_sq = torch.einsum("...i,...i->...", spheres[..., :-1], spheres[..., :-1])
    return norm_sq - 2 * spheres[..., -1]


def spheres_centers(spheres: torch.Tensor) -> torch.Tensor:
    """
    Extract the center of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (..., n) representing the center of the sphere.
    """
    return spheres[..., :-1]


# --- functions for convolution ---

def conformal_filters(spheres: Tensor, in_channels: int, kernel_size: tuple[int,int]) -> Tensor:
    """
    Transform a set of conformal spheres into 4D convolution filters.
    :param spheres: Tensor of shape (out_channels, in_channels * kh * kw + 1), representing conformal spheres.
    :param in_channels: Number of input channels for the convolution.
    :param kernel_size: Tuple (kh, kw) representing the height and width of the convolution kernel.
    :return: Tensor of shape (out_channels, in_channels, kh, kw) suitable as conv2d filters.
    """
    out_channels = spheres.shape[0]
    return spheres[:, :-1].view(out_channels, in_channels, kernel_size[0], kernel_size[1])

def conformal_einf_spheres(spheres: Tensor) -> Tensor:
    """
    Extract the e_infinite component of conformal spheres.
    :param spheres: Tensor of shape (out_channels, in_channels * kh * kw + 1), representing conformal spheres.
    :return: Tensor of shape (1, out_channels, 1, 1) containing the e_infinite components.
    """
    return spheres[:, -1].view(1, spheres.shape[0], 1, 1)

def compute_einf_tensor(
    x: Tensor,
    in_channels: int,
    kernel_size: tuple[int,int],
    stride: tuple[int,int],
    padding
) -> Tensor:
    """
    Compute 1/2 * ||x||^2 over each convolution patch of the input tensor.
    :param x: Input tensor of shape (batch size, in_channels, H, W).
    :param in_channels: Number of input channels.
    :param kernel_size: Tuple (kh, kw) representing the height and width of the convolution kernel.
    :param stride: Tuple (sh, sw) representing the stride of the convolution.
    :param padding: Padding used for the convolution ('valid', 'same', or integer).
    :return: Tensor of shape (batch size, 1, H_out, W_out) representing the 1/2 * ||x||^2 for each patch.
    """
    x_norm = F.conv2d(
        input=torch.square(x),
        weight=torch.ones([1, in_channels, kernel_size[0], kernel_size[1]], device=x.device, dtype=x.dtype),
        stride=stride,
        padding=padding
    )
    return 0.5 * x_norm