import einops
import torch


class Perception(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        Ki = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Klap = 0.25 * torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]])

        self.register_buffer("filters", torch.stack([Ki, Kx, Kx.T, Klap])[:, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[1]
        x = einops.rearrange(x, "b s h w -> (b s) () h w")
        x = torch.nn.functional.pad(x, [1, 1, 1, 1], "circular")  # Keep same size
        x = torch.nn.functional.conv2d(x, self.filters)
        x = einops.rearrange(x, "(b s) k h w -> b (k s) h w", k=4, s=s)
        return x


class Evolution(torch.nn.Module):
    def __init__(self, ns: int, ng: int) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(4 * ns + ng, 2 * 4 * ns, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(2 * 4 * ns, ns, kernel_size=1, bias=False)
        self.conv2.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ds = self.conv2(torch.relu(self.conv1(x)))
        return ds


class CellularAutomata(torch.nn.Module):
    def __init__(self, ns: int, ng: int) -> None:
        super().__init__()

        self.perception = Perception()
        self.evolution = Evolution(ns, ng)

    def forward(self, s: torch.Tensor, xg: torch.Tensor) -> torch.Tensor:
        xs = self.perception(s)
        x = torch.cat([xs, xg], dim=1)
        ds = self.evolution(x)
        update_mask = torch.randn_like(x[:, :1, :, :]) > 0

        return s + ds * update_mask
