import pytest
import torch

from hyperspherical.initializers.spherical_init import kmeans_
from hyperspherical.utils import spheres_radii


def test_kmeans():
    data = torch.randn(100, 3)
    spheres = torch.empty(5, 4)
    assert kmeans_(spheres, data)
    centers = spheres[:, :-1]
    radii = spheres_radii(spheres)
    assert torch.isfinite(centers).all()
    assert (radii > 0).all()


def test_kmeans_deterministic_with_seed():
    torch.manual_seed(42)
    data = torch.randn(50, 2)
    spheres = torch.empty(3, 3)
    _ = kmeans_(spheres, data, random_state=0)
    c1 = spheres.clone()
    _ = kmeans_(spheres, data, random_state=0)
    assert torch.allclose(c1, spheres)


@pytest.mark.parametrize("n_clusters, n_samples", [(5, 200), (10, 500)])
def test_kmeans_number_of_clusters(n_clusters, n_samples):
    data = torch.randn(n_samples, 4)
    spheres = torch.empty(n_clusters, 5)
    _ = kmeans_(spheres, data)
    assert spheres.size(0) == n_clusters
