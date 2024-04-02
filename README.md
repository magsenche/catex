# catex
This project aims to generate textures using Neural Cellular Automata (NCA).

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
