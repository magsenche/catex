import torch


class Goal(torch.nn.Module):
    def __init__(self, n_goal: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(n_goal, embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        x = self.embedding(i)
        x = self.ffwd(self.norm(x)) + x
        return x
