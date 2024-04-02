import pathlib

import matplotlib.pyplot as plt
import sklearn.manifold
import torch

import catex


def tsne(g: torch.Tensor, imgs: list, save_dir: pathlib.Path) -> None:
    tsne_map = sklearn.manifold.TSNE(n_components=2, perplexity=4, random_state=42)
    embeddings_2d = tsne_map.fit_transform(g.squeeze(1).cpu().numpy())

    plt.figure()
    catex.display.plot_thumbnails(plt.gca(), imgs, embeddings_2d, 0.5)
    plt.xlim(embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1)
    plt.ylim(embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1)
    plt.title("t-SNE")
    plt.savefig(save_dir / "tnse.png")
    plt.show()
