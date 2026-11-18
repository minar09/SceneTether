#!/usr/bin/env bash
# SceneTether inference with the exact settings used in the paper.
#
#   bash scripts/run_scenetether.sh <input_folder> <output_folder> [seconds=60] [seed=42]
#
# <input_folder> holds target_crop_info_26-15.json and the images in 26-15/ (see tools/make_input_folder.py).
# Each block generates one latent = 4 frames at 16 fps, so 60 s = 240 latents and 961 decoded frames.
# Per-video seed = seed + index of the image in the input file (the paper used seed 42 and 1042).
set -euo pipefail
IN=${1:?input folder}; OUT=${2:?output folder}; SEC=${3:-60}; SEED=${4:-42}
LAT=$((SEC * 4))
export SNF_PER_PROMPT_SEED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=${CFPP_CKPT:-checkpoints/causal-forcing++/framewise-2step.pt}
python inference.py \
  --config_path configs/cfpp_phase1_60s.yaml --phase1_config configs/phase1/echo_forcing.yaml \
  --phase1_set num_frame_per_block=1 \
  --phase1_set factorial.echo_pin_source=true \
  --phase1_set factorial.forget.roles=online_var \
  --phase1_set factorial.forget.rho=0.3 \
  --phase1_set factorial.forget.recent_s=0.0 \
  --phase1_set factorial.forget.rule=keynorm \
  --phase1_set factorial.forget.role_older=false \
  --phase1_set factorial.forget.prev_split=true \
  --phase1_set factorial.forget.prev_static_frac=0.3 \
  --phase1_set factorial.forget.prev_band=0.0 \
  --phase1_set factorial.forget.prev_freeze=true \
  --phase1_set factorial.latent_indexed_renoise=true \
  --phase1_set factorial.context_accounting=true \
  --phase1_set eval_first_n=${EVAL_FIRST_N:-64} \
  --force_num_frames "$LAT" --num_output_frames "$((LAT + 1))" \
  --output_folder "$OUT" --checkpoint_path "$CKPT" --data_path "$IN" \
  --i2v --use_ema --seed "$SEED" \
  --phase1_set factorial.forget.prev_tether=1.0 \
  --phase1_set factorial.forget.prev_tether_band=0.5
