import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import KMeans



def default_(spheres: Tensor,
             data: Tensor | np.ndarray,
             **kwargs) -> Tensor:
    """
    Default initialization method for spheres.
    Initializes spheres with random centers and unit with propagation garanteed.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples.
    :param kwargs: Additional arguments
    :return: Initialized spheres tensor of shape (k, n + 1).
    """
    n = spheres.size(1) - 1  # dimension of the spheres
    j = spheres.size(0)      # number of spheres

    #TODO: La méthode d'initialisation intelligente ici
    return spheres

def kmeans_(spheres: Tensor,
            data: Tensor | np.ndarray,
            **kwargs) -> Tensor:
    """
    KMeans initialization method for spheres.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples.
    """
    if data is None:
        raise ValueError("Data must be provided for KMeans initialization.")
    elif isinstance(data, Tensor) and data.numel() == 0:
        raise ValueError("Data must be provided for KMeans initialization.")
    elif isinstance(data, np.ndarray) and data.size == 0:
        raise ValueError("Data must be provided for KMeans initialization.")
    elif data.dim() != 2:
        raise ValueError("Data must be a 2D Tensor with shape (n_samples, n_features).")
    elif isinstance(data, Tensor):
        data = data.detach().cpu().numpy()
    assert isinstance(data, np.ndarray)
    j = spheres.size(0)
    n = spheres.size(1) - 1
    if data.shape[1] != n:
        raise ValueError(f"Data features must match the sphere dimensions: {data.shape[1]} != {n}.")
    is_requires_grad = spheres.requires_grad

    Kmeans = KMeans(n_clusters=j, **kwargs).fit(data)
    centers = Kmeans.cluster_centers_
    labels = Kmeans.labels_
    radii = np.zeros(j)
    for i in range(j):
        cluster_points = data[labels == i]
        # TODO: change the radius calculation to ensure non overlap
        if cluster_points.shape[0] > 0:
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            radii[i] = np.max(distances)
        else:
            radii[i] = 1.0  # Default radius if no points in cluster
    spheres[:, :-1] = torch.tensor(centers, dtype=spheres.dtype, device=spheres.device)
    spheres[:, -1] = 0.5 * (
            torch.sum(spheres[:, :-1] ** 2, dim=1) -
            torch.tensor(radii ** 2, dtype=spheres.dtype, device=spheres.device)
    )
    spheres.requires_grad = is_requires_grad
    return True


def random_(spheres: Tensor,
            data: Tensor | np.ndarray = None,
            radius: float = 1.0,
            **kwargs) -> Tensor:
    """
    Random initialization method for spheres.
    Initializes spheres with random centers and specified radius.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples. (Not used here)
    :param radius: Radius to initialize the spheres with.
    :param kwargs: Additional arguments
    :return: Initialized spheres tensor of shape (k, n + 1).
    """
    n = spheres.size(1) - 1  # dimension of the spheres
    j = spheres.size(0)      # number of spheres

    centers = torch.randn(j, n, dtype=spheres.dtype, device=spheres.device)
    radii = torch.full((j,), radius, dtype=spheres.dtype, device=spheres.device)

    spheres[:, :-1] = centers
    spheres[:, -1] = 0.5 * (torch.sum(centers ** 2, dim=1) - radii ** 2)
    return True
