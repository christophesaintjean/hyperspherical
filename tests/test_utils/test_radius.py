import torch

from hyperspherical.utils import (
    conformal_point,
    conformal_sphere,
    sphere_squared_radius,
    spheres_radii,
)


def test_sphere_radius():
    centers = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-1.0, 3.0]])
    radii = torch.tensor([1.0, 2.0, 3.0, 10.0])
    spheres = conformal_sphere(centers, radii)
    computed_radii = spheres_radii(spheres)
    assert torch.allclose(computed_radii, radii, atol=1e-5)


def test_sphere_squared_radius():
    centers = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-1.0, 3.0]])
    radii = torch.tensor([1.0, 2.0, 3.0, 10.0])
    spheres = conformal_sphere(centers, radii)
    computed_radii = sphere_squared_radius(spheres)
    assert torch.allclose(computed_radii, torch.square(radii), atol=1e-5)


def test_sphere_radius_zero_radius():
    centers = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    radii = torch.tensor([0.0, 0.0])
    spheres = conformal_sphere(centers, radii)
    computed_radii = spheres_radii(spheres)
    assert torch.allclose(computed_radii, radii, atol=1e-5)

    points = conformal_point(centers)
    assert torch.allclose(points, spheres, atol=1e-5)
