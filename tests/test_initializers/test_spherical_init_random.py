import torch

from hyperspherical.initializers.spherical_init import random_


def test_random():
    spheres = torch.empty(5, 4)
    assert random_(spheres, radius=2.0)
    centers = spheres[:, :-1]
    radii = torch.sqrt(torch.sum(centers**2, dim=1) - 2 * spheres[:, -1])
    assert torch.isfinite(centers).all()
    torch.testing.assert_close(radii, 2.0 * torch.ones_like(radii))
