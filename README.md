# SceneTether: Activity-Gated Value Anchoring for Long-Horizon Autoregressive Image Animation

[Matiur Rahman Minar](https://minar09.github.io/) · [Seunghun Oh](https://owinhun.github.io/) · [GangHyeon Jeong](https://github.com/Jugahy) · [Unsang Park](https://cviplab.sogang.ac.kr/cviplab/1249.html)

Sogang University

[![arXiv](https://img.shields.io/badge/arXiv-2611.28134-b31b1b.svg)](https://arxiv.org/abs/2611.28134)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://minar09.github.io/scenetether/)

**Official codebase release** for the paper [*SceneTether: Activity-Gated Value Anchoring for Long-Horizon Autoregressive Image Animation*](https://arxiv.org/abs/2611.28134) ([PDF](https://arxiv.org/pdf/2611.28134)). Visit the [project page](https://minar09.github.io/scenetether/) for visual results and more information.

## Overview

Training-free long-horizon image animation for fixed-camera nature scenes.

SceneTether changes how a causal (autoregressive) video diffusion model reads its most recent frame. It keeps every
token and key of the newest latent. At low-activity positions, it blends the values toward the source image's values
at the same positions; high-activity positions keep their recent values so the motion can evolve. An activity map
estimated from the first six generated latents (1.5 s) and then frozen decides where to blend. The method adds no
denoising evaluations and leaves the cache-update rule unchanged.

This repository runs SceneTether on the frame-wise two-step **Causal Forcing++** student of Wan2.1-T2V-1.3B,
hosted in an Echo-Forcing memory with the source latent pinned, exactly as evaluated in the paper.

## Setup

Tested with Python 3.10, CUDA 12.1 and one NVIDIA A100 80 GB.

```bash
pip install -r requirements.txt          # flash-attn may need: pip install flash-attn==2.8.3 --no-build-isolation
```

Download the base model and the Causal Forcing++ student:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
```

`WAN_MODELS_ROOT` (default `wan_models`) and `CFPP_CKPT` (default
`checkpoints/causal-forcing++/framewise-2step.pt`) override these locations.

## Inputs

An input folder holds `target_crop_info_26-15.json` and the 832x480 images in `26-15/`. Build one from your own
images and prompts:

```bash
python tools/make_input_folder.py --out inputs/demo \
  --pair my_photo.jpg "$(cat examples/prompt_tropical_shore.txt)"
```

The model works best with a fixed-camera photograph of moving natural material (water, waves, smoke, steam, fire,
dust, rain or snow) and a prompt that says the camera is fixed, names what stays still and describes what moves.
`examples/` contains prompts in that style.

## Run

```bash
bash scripts/run_scenetether.sh inputs/demo outputs/demo_60s 60      # 60 s
bash scripts/run_scenetether.sh inputs/demo outputs/demo_240s 240    # four minutes
```

The arguments are the input folder, the output folder, the duration in seconds (default 60) and the base seed
(default 42). Each image gets seed `base + index`. Output videos are 832x480 at 16 fps, written as
`<first 100 characters of the prompt>.mp4`, next to a `*_forget_map.png` that shows the frozen activity map and a
`*_context.json` that records what each block read.

On one A100, generation runs at about 7 frames per second (about 5.5 with decoding), roughly 1% slower than the same
host without SceneTether, and peaks at about 28 GiB of GPU memory. With less than 40 GiB of free GPU memory, the
inference script switches to a low-memory model-loading path; for bit-exact reproduction of our videos, run with at
least 40 GiB free.

## What the settings mean

`scripts/run_scenetether.sh` passes the paper's settings explicitly:

| Setting | Value | Meaning |
|---|---|---|
| `num_frame_per_block` | 1 | one latent (4 frames, 0.25 s) per block |
| `echo_pin_source` | true | the source latent stays in the read for the whole rollout |
| `forget.roles=online_var`, `prev_freeze` | | activity map from the variance of the first 6 generated latents, Otsu threshold, then frozen |
| `forget.prev_tether` | 1.0 | blend strength in the low-activity interior (graded by a taper of width 0.3) |
| `forget.prev_tether_band` | 0.5 | blend strength on the one-token band around the high-activity region |
| `forget.rho` | 0.3 | older history latents keep round(1560 * 0.3^d) tokens at age d |
| `latent_indexed_renoise` | true | each re-noising draw is seeded by (seed + image index, latent index, timestep) |

The code contains other options used for comparisons and controls in the paper; they are off by default and are not
needed to run SceneTether.

## License and acknowledgements

Apache-2.0 (see `LICENSE`). This code builds on [Causal Forcing](https://github.com/thu-ml/Causal-Forcing),
[Self Forcing](https://github.com/guandeh17/Self-Forcing), CausVid and [Wan2.1](https://github.com/Wan-Video/Wan2.1);
files under `wan21/` are modified from Wan2.1 and keep their original copyright notices. The Echo memory follows
the Echo-Forcing paper. Model weights are subject to their own licenses.

## Citation

If you use this code or find SceneTether useful in your research, please cite our paper:

```bibtex
@article{minar2026scenetether,
  title         = {SceneTether: Activity-Gated Value Anchoring for Long-Horizon Autoregressive Image Animation},
  author        = {Matiur Rahman Minar and Seunghun Oh and Ganghyeon Jeong and Unsang Park},
  journal       = {arXiv preprint arXiv:2611.28134},
  year          = {2026},
  eprint        = {2611.28134},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2611.28134},
  url           = {https://arxiv.org/abs/2611.28134}
}
```
