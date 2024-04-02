import pathlib

import einops
import torch
import torchvision

import catex

REPEAT_PATTERN = "b g -> b g h w"


## Gifs for each textures
def texture(
    a: int,
    ca: torch.nn.Module,
    g: torch.Tensor,
    shape: tuple[int],
    exp_dir: pathlib.Path,
    n_step: int = 2000,
) -> None:
    s, h, w = shape
    ga = g[a]
    xg = einops.repeat(ga, REPEAT_PATTERN, h=h, w=w)
    st = torch.zeros((1, s, h, w))
    run_exp(ca, st, xg, exp_dir / f"{a}.gif", n_step)


## Gifs for interpolation between two textures
def interpolation(
    a: int,
    b: int,
    p: float,
    ca: torch.nn.Module,
    g: torch.Tensor,
    shape: tuple[int],
    exp_dir: pathlib.Path,
    n_step: int = 2000,
) -> None:
    s, h, w = shape
    gp = p * g[a] + (1 - p) * g[b]
    xg = einops.repeat(gp, REPEAT_PATTERN, h=h, w=w)
    st = torch.zeros((1, s, h, w))
    run_exp(ca, st, xg, exp_dir / f"{a}{b}_{p}.gif", n_step)


## Gifs for each pair of textures
def mix(
    a: int,
    b: int,
    ca: torch.nn.Module,
    g: torch.Tensor,
    shape: tuple[int],
    exp_dir: pathlib.Path,
    n_step: int = 2000,
) -> None:
    s, h, w = shape
    gab = 0.5 * (g[a] + g[b])
    xg = einops.repeat(gab, REPEAT_PATTERN, h=h, w=w)
    st = torch.zeros((1, s, h, w))
    run_exp(ca, st, xg, exp_dir / f"{a}{b}.gif", n_step)


## Gifs for each pair of textures side by side
def sbs(
    a: int,
    b: int,
    ca: torch.nn.Module,
    g: torch.Tensor,
    shape: tuple[int],
    exp_dir: pathlib.Path,
    n_step: int = 2000,
) -> None:
    s, h, w = shape
    ga = g[a]
    gb = g[b]
    xga = einops.repeat(ga, REPEAT_PATTERN, h=h, w=w // 2)
    xgb = einops.repeat(gb, REPEAT_PATTERN, h=h, w=w // 2)
    xg = torch.cat([xga, xgb], dim=-1)
    st = torch.zeros((1, s, h, w))
    run_exp(ca, st, xg, exp_dir / f"{a}{b}.gif", n_step)


## Gifs for detection
def detect(
    a: int,
    samples: torch.Tensor,
    ca: torch.nn.Module,
    g: torch.Tensor,
    s: int,
    exp_dir: pathlib.Path,
    n_step: int = 2000,
) -> None:
    s0 = einops.rearrange(samples, "n c h w -> c h (n w)")
    h, w = s0.shape[1:]
    ga = g[a]
    xg = einops.repeat(ga, "b g -> b g h w", h=h, w=w)
    st = torch.zeros((1, s, h, w))
    st[0, :3] = s0.unsqueeze(0) - 0.5  # rgb to nca_rgb
    catex.experiment.generate.run_exp(ca, st, xg, exp_dir / f"{a}.gif", n_step)


def run_exp(
    ca: torch.nn.Module,
    st: torch.Tensor,
    xg: torch.Tensor,
    exp_name: pathlib.Path,
    n_step: int,
) -> None:
    with torch.no_grad():
        imgs = []
        for i in range(n_step):
            if i % (n_step // 100) == 0:
                imgs.append(torchvision.transforms.ToPILImage()(st[0, :3] + 0.5))
            st = ca(st, xg)

    catex.display.save_gif(imgs, exp_name)
