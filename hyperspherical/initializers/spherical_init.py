import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor


def default_(spheres: Tensor, data: Tensor | np.ndarray, **kwargs) -> bool:
    """
    Default initialization method for spheres.
    Initializes spheres with random centers and unit with propagation garanteed.
    :param spheres: Tensor of shape (k, n + 1) where k is the number of spheres and n is the dimension.
    :param data: Input data tensor of shape (m, n) where m is the number of samples.
    :param kwargs: Additional arguments
    :return: Whether or not the spheres were initialized.

    """
    # TODO: La méthode d'initialisation intelligente ici
    # n = spheres.size(1) - 1  # dimension of the spheres
    # j = spheres.size(0)  # number of spheres
    return False


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


def sph_init_methode(spheres: Tensor, data: Tensor | np.ndarray, **kwargs) -> tuple[bool, dict]:
    """
    Initialization of spheres using the proposed methode
    :param spheres: tensor representing the spheres
    :param data: input data (Tensor or ndarray)
    :return: (True, dictionary containing rhos, centers and delta)
    """

    # Convert data to numpy if tensor
    data_np = data.clone().detach().cpu().numpy() if isinstance(data, Tensor) else np.asarray(data)
    mu_x, sigma_x = np.mean(data_np), np.std(data_np)
    n = data_np.shape[1]
    num_sph = spheres.size(0)

    # Constants
    eps = 1e-6
    C = (num_sph / np.sqrt(n)) - eps
    delta = np.sqrt(((num_sph**2) * n * (sigma_x**2)) / (2 * ((num_sph**2) - (C**2 * n))))

    # Upper bound for mu_s (same for all spheres)
    upper_mu_s1 = mu_x + np.sqrt((delta**2 / n) - (sigma_x**2 / 2))
    upper_mu_s2 = mu_x + (delta / num_sph)
    mu_s = min(upper_mu_s1, upper_mu_s2) - eps

    # sigma_s² and rhos
    sig_s2 = -((mu_s - mu_x) ** 2 + sigma_x ** 2) + np.sqrt((mu_x - mu_s) ** 4 + (2 * (delta ** 2 / n) * sigma_x ** 2))
    rhos = np.sqrt(n * (sigma_x ** 2 + sig_s2 + (mu_x - mu_s) ** 2) + 2 * delta * mu_x)

    # Generate centers c (Gaussian sampling)
    cs = np.random.normal(mu_s, np.sqrt(sig_s2), (num_sph, n))

    return True, {
        "rhos": torch.tensor(rhos, dtype=torch.float32),
        "centers": torch.tensor(cs, dtype=torch.float32),
        "delta": delta,
    }

