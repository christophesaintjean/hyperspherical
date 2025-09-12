import torch
from torch import nn, Tensor
from typing import Callable
from hyperspherical.utils import conformal_filters, conformal_einf_spheres, compute_einf_tensor

class Conv2dSpherical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int,int] = 3,
        stride: int | tuple[int,int] = 1,
        padding: str | int | tuple[int,int] = "valid",
        init_method: str | Callable | None = None,
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


    def forward(self, x: Tensor) -> Tensor:
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

        return out / self.delta
