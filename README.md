# Runtime Failure Detection for Vision-Language-Action Models by Probing Internal Model States

## Installation

When cloning this repo, make sure to update submodules:

```
git clone --recurse-submodules git@github.com:ravtscheev/runtime-failure-detection.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

## Repository Structure

- `configs/` – configs for all experiment phases
- `src/` – core pipeline (simulation -> training -> safety)
- `notebooks/` – analysis used in the thesis
- `scripts/` – end-to-end experiment runners

# Data Generation
Goal: Create large-scale synthetic data from a few human demos.

First we need to install the dependencies:
```
uv pip install -r requirements-local.txt
```

## Convert HDF5 to LeRobot Format
To convert LeRobot HDF5 datasets into the required format, provide the necessary configuration in `configs/lerobot_convert.yaml` and then run:

```
uv run scripts/convert_hdf5_lerobot.py --config configs/lerobot_convert.yaml
```

# Acknowledgements

This project builds on open-source work including:
- MimicGen

I am grateful to the authors and maintainers of these projects.

