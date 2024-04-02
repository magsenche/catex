import torch
import torchvision


def get_texture_loss(dataset_sample):
    target = torch.stack(dataset_sample)

    style_params = {
        "model": torchvision.models.vgg16(weights="IMAGENET1K_V1").features,
        "layers": [1, 6, 11, 18, 25],
        "transform": torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    }

    target_styles = vgg_styles(target, style_params)

    def style_loss(x: torch.Tensor, n: int) -> torch.Tensor:
        x_styles = vgg_styles(x, style_params)
        return sum(
            opt_transport_loss(x, y[n].unsqueeze(0))
            for (x, y) in zip(x_styles, target_styles)
        )

    return style_loss


def vgg_styles(x: torch.Tensor, params: dict) -> list[torch.Tensor]:
    # Extract features from a tensor
    x = params["transform"](x)
    features = [x]
    for i, layer in enumerate(params["model"][: max(params["layers"]) + 1]):
        x = layer(x)
        if i in params["layers"]:
            features.append(x)
    return features


def opt_transport_loss(
    source: torch.Tensor, target: torch.Tensor, proj_dim: int = 32
) -> torch.Tensor:
    # Sliced wasserstein loss (optimal transport) https://arxiv.org/pdf/2006.07229.pdf
    source = source.flatten(-2)
    target = target.flatten(-2)
    c, f = source.shape[-2:]
    style_w = torch.nn.functional.normalize(torch.randn(c, proj_dim), dim=0)
    source_proj = torch.einsum("bcf,cp->bpf", source, style_w).sort()[0]
    target_proj = torch.einsum("bcf,cp->bpf", target, style_w).sort()[0]
    target_proj = torch.nn.functional.interpolate(target_proj, f, mode="nearest")
    loss = (source_proj - target_proj.detach()).square().sum()
    return loss


def overflow_loss(x: torch.Tensor) -> torch.Tensor:
    loss = (x - x.clamp(-1.0, 1.0)).abs().sum()
    return loss
