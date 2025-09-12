import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

def default_(spheres: Tensor, data: Tensor | np.ndarray, eps: float = 1e-6, **kwargs) -> bool:
    """
    Default initialization method for spheres.
    Initializes spheres with random centers and unit with propagation garanteed.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples.
    :param kwargs: Additional arguments
    :return: Whether or not the spheres were initialized.

    """
    if data is None:
        raise ValueError("Data must be provided for initialization.")
    elif isinstance(data, Tensor) and data.numel() == 0:
        raise ValueError("Data must be provided for initialization.")
    elif isinstance(data, np.ndarray) and data.size == 0:
        raise ValueError("Data must be provided for initialization.")
    elif (isinstance(data, Tensor) and data.dim() != 2) or (isinstance(data, np.ndarray) and data.ndim != 2):
        raise ValueError("Data must be 2D (n_samples, n_features).")

    if isinstance(data, Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data

    assert isinstance(data_np, np.ndarray)

    n = spheres.size(1) - 1  # dimension of the spheres
    j = spheres.size(0)  # number of spheres
    if data_np.shape[1] != n:
        raise ValueError(f"Data features must match the sphere dimensions: {data_np.shape[1]} != {n}")

    # Required buffer delta
    if "delta" not in kwargs:
        raise ValueError("delta must be passed via kwargs.")
    delta_buf: Tensor = kwargs["delta"]

    mu_x, sigma_x = np.mean(data_np), np.std(data_np)

    # Constants
    C = (j / np.sqrt(n)) - eps
    delta = np.sqrt(((j**2) * n * (sigma_x**2)) / (2 * ((j**2) - (C**2 * n))))

    # Upper bound for mu_s (same for all spheres)
    upper_mu_s1 = mu_x + np.sqrt((delta**2 / n) - (sigma_x**2 / 2))
    upper_mu_s2 = mu_x + (delta / j)
    mu_s = min(upper_mu_s1, upper_mu_s2) - eps

    # sigma_s² and rhos
    sig_s2 = -((mu_s - mu_x) ** 2 + sigma_x ** 2) + np.sqrt((mu_x - mu_s) ** 4 + (2 * (delta ** 2 / n) * sigma_x ** 2))
    rhos = np.sqrt(n * (sigma_x ** 2 + sig_s2 + (mu_x - mu_s) ** 2) + 2 * delta * mu_x)

    # Generate centers c (Gaussian sampling)
    cs = np.random.normal(mu_s, np.sqrt(sig_s2), (j, n))

    # Fill spheres and update delta in place (torch only)
    with torch.no_grad():
        centers_t = torch.tensor(cs, dtype=spheres.dtype, device=spheres.device)
        rhos_t = torch.tensor(rhos, dtype=spheres.dtype, device=spheres.device)

        spheres[:, :-1] = centers_t
        p_squared = (spheres[:, :-1] ** 2).sum(dim=-1)
        spheres[:, -1] = 0.5 * (p_squared - rhos_t ** 2)

        delta_buf.copy_(torch.tensor(delta, dtype=spheres.dtype, device=spheres.device))

    return True



def kmeans_(spheres: Tensor, data: Tensor | np.ndarray, **kwargs) -> bool:
    """
    KMeans initialization method for spheres.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples.
    :param kwargs: Additional arguments for sklearn KMeans.
    :return: Whether or not the spheres were initialized.
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
        torch.sum(spheres[:, :-1] ** 2, dim=1)
        - torch.tensor(radii**2, dtype=spheres.dtype, device=spheres.device)
    )
    spheres.requires_grad = is_requires_grad
    return True


def random_(
    spheres: Tensor, data: Tensor | np.ndarray = None, radius: float = 1.0, **kwargs
) -> bool:
    """
    Random initialization method for spheres.
    Initializes spheres with random centers and specified radius.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples. (Not used here)
    :param radius: Radius to initialize the spheres with.
    :param kwargs: Additional arguments
    :return: Whether or not the spheres were initialized.
    """
    n = spheres.size(1) - 1  # dimension of the spheres
    j = spheres.size(0)  # number of spheres

    centers = torch.randn(j, n, dtype=spheres.dtype, device=spheres.device)
    radii = torch.full((j,), radius, dtype=spheres.dtype, device=spheres.device)

    spheres[:, :-1] = centers
    spheres[:, -1] = 0.5 * (torch.sum(centers**2, dim=1) - radii**2)
    return True

