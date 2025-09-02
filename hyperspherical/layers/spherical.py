import torch
import torch.nn as nn
from torch import Tensor
import logging


from hyperspherical.utils import conformal_point, conformal_product
from hyperspherical.initializers import default_


class Spherical(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    spheres: Tensor
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_method: str | callable | None = default_,
                 init_args: dict | None = None,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Spherical, self).__init__(**factory_kwargs)
        self.in_features = in_features    # dimension of spheres
        self.out_features = out_features  # number of spheres
        self.spheres = nn.Parameter(
            torch.empty(self.out_features, self.in_features + 1, **factory_kwargs),
            requires_grad=True
        )
        self.init_method = init_method
        self.init_args = init_args if init_args is not None else {}
        self.initialized = False
        self._initialize_spheres()


    def _initialize_spheres(self, x: torch.Tensor) -> None:
        """
        Initialize the spherical layer.
        :param x: Input tensor to determine the shape of the spheres.
        """
        if self.initialized:
            # already initialized -> warning
            logging.warn("Spherical layer already initialized. Re-initialization skipped.")
            return
        assert(callable(self.init_method))
        res = self.init_method(self.spheres, x, **self.init_args)
        self.initialized = res is not None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the spherical layer.
        :param x: Input tensor of shape (batch_size, in_features).
        :return: Output tensor of shape (batch_size, out_features).
        """
        if not self.initialized:
            self._initialize_spheres(x)

        x_tilde = conformal_point(x)
        return self.conformal_product(self.spheres, x_tilde)


