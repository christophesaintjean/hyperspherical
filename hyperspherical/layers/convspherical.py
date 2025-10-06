import torch
import logging

from torch import nn, Tensor
from typing import Callable
from hyperspherical.utils import conformal_filters, conformal_einf_spheres, compute_einf_tensor
from hyperspherical.initializers import default_

class Conv2dSpherical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int,int] = 3,
        stride: int | tuple[int,int] = 1,
        padding: str | int | tuple[int,int] = "same",
        init_method: str | Callable | None = default_,
        init_args: dict | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dim = in_channels * kernel_size[0] * kernel_size[1]

        # Paramètres conformes des sphères
        self.spheres = nn.Parameter(
            torch.empty(out_channels, self.dim + 1, **factory_kwargs),
            requires_grad=True
        )

        self.init_method = init_method
        self.init_args = init_args or {}
        self.initialized = False
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
        if not self.initialized:
            self._initialize_spheres(x.clone().view(x.size(0), -1))


        conv_sx = torch.nn.functional.conv2d(
            input=x,
            weight=conformal_filters(self.spheres, self.in_channels, self.kernel_size),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=1
        )
        out =  conv_sx - conformal_einf_spheres(self.spheres) - compute_einf_tensor(
            x, self.in_channels, self.kernel_size, self.stride, self.padding
        )

        return out / self.delta**2
