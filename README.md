# catex
This project aims to generate textures using Neural Cellular Automata (NCA). It is heavily inspired by [Self-Organising Textures](https://distill.pub/selforg/2021/textures/)

It uses vgg style loss with [a sliced Wasserstein loss for neural texture synthesis](https://arxiv.org/pdf/2006.07229.pdf)

## Setup
1. Install dependencies from `pyproject.toml` e.g. using pdm: `pdm install`
2. Set up environment variables: `BCHW`, `G`, `N_EPOCH`, `N_EXP`, `DATASET_DIR`, `MODEL_DIR`, `CHECKPOINT_PATH` (see [notebook](demo/poc.ipynb))
    - if you are using pdm, just push them in an `.env` at the project root

## Proof-of-concept
The [notebook](demo/poc.ipynb) demonstrates the proof-of-concept of the texture generation process.

The Neural Cellular Automata (NCA) is a dynamic algorithm where the state of cells constantly updates. The result represents the evolution of cells from an initial state to the desired target. The output is upscaled (4x) to demonstrate its robustness to scale.

**Target**

![](assets/poc/target.png)

**Result**

![](assets/poc/result.gif)

## Goal oriented CA
Overcome Cellular Automata limitation to generate a unique target (the one it was trained on) by adding dedicated hidden channels that are optimized for a specific target.

The resulting values forms a **footprint** of a pattern: by initializing different footprint on a same grid, a single cellular automata can generate different patterns.

You can also consider mixing or interpoling footprint for **zero-shot texture generation**.

Check the goal [experiment script](experiment/main.py) and the [notebook](demo/goal.ipynb)
