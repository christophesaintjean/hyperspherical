import pytest
import torch

from hyperspherical.initializers.spherical_init import kmeans_

def test_kmeans_tensor_input_shapes():
    data = torch.randn(100, 3)
    spheres = torch.empty(5, 4)
    centers = kmeans_(spheres, data)

    assert centers.shape == spheres.shape
    assert torch.isfinite(centers).all()

def test_kmeans_deterministic_with_seed():
    torch.manual_seed(42)
    data = torch.randn(50, 2)
    spheres = torch.empty(3, 3)
    c1 = kmeans_(spheres, data)

    torch.manual_seed(42)
    c2 = kmeans_(spheres, data)

    assert torch.allclose(c1, c2)

@pytest.mark.parametrize("n_clusters, n_samples", [(5, 200), (10, 500)])
def test_kmeans_number_of_clusters(n_clusters, n_samples):
    data = torch.randn(n_samples, 4)
    spheres = torch.empty(n_clusters, 5)
    centers = kmeans_(spheres, data)

    assert centers.shape[0] == n_clusters
