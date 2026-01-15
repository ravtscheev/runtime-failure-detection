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

If you want to upload the converted files to HuggingFace, you first need to login:

```
hf auth login
```

Uploading a dataset after conversion can be done by:

```
hf upload {path/to/dataset_folder} --repo-id {username/my-cool-dataset} --repo-type dataset
```

# Acknowledgements

This project builds on open-source work including:
- MimicGen
- LeRobot

I am grateful to the authors and maintainers of these projects.

