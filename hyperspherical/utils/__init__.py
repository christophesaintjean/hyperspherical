import torch

__all__ = [
    'conformal_point',
    'conformal_sphere',
    'conformal_product',
    'sphere_radius',
    'sphere_center'
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

def conformal_center(sphere: torch.Tensor) -> torch.Tensor:
    """
    Extract the center of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (..., n+1) representing the center of the sphere.
    """
    return conformal_point(sphere[..., :-1])

def conformal_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the conformal product of two tensors.
    :param x: Tensor of shape (..., n + 1), where n is the dimension of the input.
    :param y: Tensor of shape (..., n + 1).
    :return: Tensor of shape (..., n) representing the conformal product.
    """
    return torch.einsum('...si,...bi->...bs', x[..., :-1], y[..., :-1]) - x[..., -1].view(1, -1) - y[..., [-1]]

def sphere_radius(spheres: torch.Tensor) -> torch.Tensor:
    """
    Compute the radius of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (...) representing the radius of the sphere.
    """
    #TODO: optimize this function with einsum
    return torch.sqrt(torch.diag(conformal_product(spheres, spheres)))

def sphere_center(spheres: torch.Tensor) -> torch.Tensor:
    """
    Extract the center of a conformal sphere.
    :param sphere: Tensor of shape (..., n + 1) representing the conformal sphere.
    :return: Tensor of shape (..., n) representing the center of the sphere.
    """
    return spheres[..., :-1]