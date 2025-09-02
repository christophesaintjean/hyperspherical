import torch
from hyperspherical.utils import sphere_radius, conformal_sphere


def test_sphere_radius():
    centers = torch.tensor([[0.0, 0.0],
                            [1.0, 1.0],
                            [2.0, 2.0],
                            [-1.0, 3.0]])
    radii = torch.tensor([1.0, 2.0, 3.0, 10.0])
    spheres = conformal_sphere(centers, radii)
    computed_radii = sphere_radius(spheres)
    assert torch.allclose(computed_radii, radii, atol=1e-5)