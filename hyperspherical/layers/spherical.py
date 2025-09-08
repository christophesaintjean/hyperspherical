import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

from hyperspherical.initializers import default_
from hyperspherical.utils import conformal_point, conformal_product


class Spherical(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    spheres: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str | Callable | None = default_,
        init_args: dict | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.spheres = nn.Parameter(
            torch.empty(out_features, in_features + 1, **factory_kwargs),
            requires_grad=True,
        )
        self.init_method = init_method
        self.init_args = init_args or {}
        self.initialized = False

        # delta is fixed at initialization (not trainable)
        self.register_buffer("delta", torch.tensor(1.0, **factory_kwargs))

    def _initialize_spheres(self, x: Tensor) -> None:
        if self.initialized:
            logging.warning("Spherical layer already initialized. Re-initialization skipped.")
            return

        if not callable(self.init_method):
            raise TypeError("init_method must be callable.")

        res = self.init_method(self.spheres, x, delta=self.delta, **self.init_args)

        if not res:
            raise RuntimeError("Initialization method failed.")

        self.initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the spherical layer.
        :param x: Input tensor of shape (batch_size, in_features).
        :return: Output tensor of shape (batch_size, out_features).
        """
        if not self.initialized:
            self._initialize_spheres(x)

        x_tilde = conformal_point(x)
        out = conformal_product(self.spheres, x_tilde)
        return out / self.delta
