import itertools
import os
import pathlib

import einops
import numpy
import torch
import torchvision
import tqdm

import catex


def train(
    ca: catex.model.CellularAutomata,
    goal: catex.model.Goal,
    shape: tuple[int],
    loss_f: callable,
    n_exp: int,
    n_epoch: int,
    save_dir: pathlib.Path,
):
    optimizer = torch.optim.Adam(ca.parameters(), lr=1e-3, capturable=True)

    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)

    g_optimizer = torch.optim.Adam(goal.parameters(), lr=1e-3, capturable=True)

    glr_sched = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, [1000, 2000], 0.3)

    b, s, h, w = shape

    pool = torch.zeros((n_exp, b, s, h, w))
    for i in tqdm.trange(n_epoch):
        for n in range(n_exp):
            g = torch.Tensor(4 * [n]).to(torch.int64)
            xg = einops.repeat(goal(g), "b g -> b g h w", h=h, w=w)

            batch_idx = numpy.random.choice(b, 4, replace=False)
            x = pool[n, batch_idx]
            if i % 8 == 0:
                x[:1] = 0
            step_n = numpy.random.randint(32, 96)
            for k in range(step_n):
                x = ca(x, xg)

            loss = loss_f(x, n)
            loss.backward()

            for p in ca.parameters():
                p.grad /= p.grad.norm() + 1e-8
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()

            for p in goal.parameters():
                p.grad /= p.grad.norm() + 1e-8
            g_optimizer.step()
            g_optimizer.zero_grad()
            glr_sched.step()

            pool[n, batch_idx] = x.detach()
        if ((i + 1) % 500 == 0) | (i == n_epoch - 1):
            torch.save(
                {"ca": ca.state_dict(), "goal": goal.state_dict()},
                f"{save_dir}/model.pt",
            )


if __name__ == "__main__":
    numpy.random.seed(21)
    torch.manual_seed(94)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    DATASET_DIR = os.getenv("DATASET_DIR")
    MODEL_DIR = os.getenv("MODEL_DIR")
    N_EPOCH = int(os.getenv("N_EPOCH"))
    N_EXP = int(os.getenv("N_EXP"))
    b, s, h, w = tuple(eval(os.getenv("BCHW")))
    g = int(os.getenv("G"))

    save_dir = pathlib.Path(f"{MODEL_DIR}") / f"run"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)

    # Dataset
    dataset_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop((h, w))]
    )

    train_dataset = torchvision.datasets.DTD(
        root=DATASET_DIR, transform=dataset_transform, download=True
    )
    sample_idx = numpy.random.choice(len(train_dataset), N_EXP, replace=False)
    # dataset_sample = [train_dataset[i][0].to(torch.device("cuda")) for i in sample_idx]
    dataset_sample = [train_dataset[i][0] for i in sample_idx]

    tex_dir = save_dir / f"texture"
    tex_dir.mkdir(parents=True, exist_ok=True)

    style_imgs = []
    for i, x in enumerate(dataset_sample):
        style_img = torchvision.transforms.ToPILImage()(x)
        style_imgs.append(style_img)
        style_img.save(tex_dir / f"{i}.png")

    texture_loss = catex.loss.get_texture_loss(dataset_sample)
    loss_f = lambda x, n: texture_loss(x[:, :3] + 0.5, n) + catex.loss.overflow_loss(x)

    # Model
    ca = catex.model.CellularAutomata(s, g)
    goal = catex.model.Goal(N_EXP, g)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        ca.load_state_dict(checkpoint["ca"])
        goal.load_state_dict(checkpoint["goal"])
    else:
        train(ca, goal, (b, s, h, w), loss_f, N_EXP, N_EPOCH, save_dir)

    # Experiments
    tex_dir = save_dir / "texture_gen"
    tex_dir.mkdir(parents=True, exist_ok=True)

    int_dir = save_dir / "interpolation"
    int_dir.mkdir(parents=True, exist_ok=True)

    mix_dir = save_dir / "mixed"
    mix_dir.mkdir(parents=True, exist_ok=True)

    sbs_dir = save_dir / "sidebyside"
    sbs_dir.mkdir(parents=True, exist_ok=True)

    det_dir = save_dir / "detection_gen"
    det_dir.mkdir(parents=True, exist_ok=True)

    a_ids = numpy.random.choice(N_EXP, 5, replace=False)
    b_ids = numpy.random.choice(N_EXP, 5, replace=False)
    shape = (s, 2 * h, 2 * w)
    with torch.no_grad():
        g = goal(torch.Tensor([i for i in range(N_EXP)]).to(torch.int64)).unsqueeze(1)
        for a, b in itertools.product(a_ids, b_ids):
            catex.experiment.generate.mix(a, b, ca, g, shape, mix_dir)
            catex.experiment.generate.sbs(a, b, ca, g, shape, sbs_dir)
            for p in [0.2 * i for i in range(6)]:
                catex.experiment.generate.interpolation(a, b, p, ca, g, shape, int_dir)

        for a in range(N_EXP):
            catex.experiment.generate.texture(a, ca, g, shape, tex_dir)

        samples = dataset_sample[:4]
        for a in range(len(samples)):
            catex.experiment.generate.detect(a, samples, ca, g, s, det_dir)

    catex.experiment.visualize.tsne(g, style_imgs, save_dir)
