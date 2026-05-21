import torch
import torch.nn as nn


class ConvNeXt(nn.Module):
    r"""Masked 1D ConvNeXtV2 stack for text embeddings."""

    def __init__(self, dim, intermediate_dim: int | None = None, num_layers: int = 4):
        super().__init__()
        intermediate_dim = intermediate_dim or dim * 4
        self.blocks = nn.ModuleList(
            [ConvNeXtV2Block(dim=dim, intermediate_dim=intermediate_dim) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask[:, :, None], 0.0)

        for block in self.blocks:
            x = block(x)
            if mask is not None:
                x = x.masked_fill(~mask[:, :, None], 0.0)

        return x


class GRN(nn.Module):
    r"""Global response normalization from ConvNeXtV2."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x: (b, l, d)

        Returns:
            out: (b, l, d)
        """

        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return x + residual
