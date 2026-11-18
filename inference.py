import argparse
import argparse
import time
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json
import math
import re

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from scripts.make_region_mask import extract_flow_queries, generate_flow_mask_clipseg

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


# Duration marker, e.g. "...natural water flow.[60s]" or multi-action "a"[5s] | "b"[15s].
_DURATION_RE = re.compile(r'\[(\d+\.?\d*)\s*s[#]?\]')


def parse_durations_from_prompt(prompt_text):
    """Parse a target duration (in seconds) from a prompt.

    Reads duration markers like ``[60s]``. For multi-action prompts of the
    form ``"a"[5s] | "b"[15s]`` the per-action durations are summed. Anything
    after a ';' (e.g. subtitles) is ignored. Returns the total seconds, or
    ``None`` when the prompt contains no duration marker.
    """
    prompt_part = prompt_text.split(';')[0].strip()
    total_duration = 0.0
    found = False
    for scene_part in prompt_part.split('|'):
        match = _DURATION_RE.search(scene_part)
        if match:
            total_duration += float(match.group(1))
            found = True
    return total_duration if (found and total_duration > 0) else None


def strip_duration_markers(prompt_text):
    """Remove duration markers (e.g. ``[60s]``) so they are not encoded."""
    return _DURATION_RE.sub('', prompt_text).strip()


def calculate_latent_frames_from_duration(total_duration_seconds, fps, temporal_compression,
                                          num_frame_per_block, independent_first_frame,
                                          has_initial_latent):
    """Convert a target duration into a number of *noise* latent frames.

    Mirrors the pipeline's block constraints so the resulting count is always
    valid for ``CausalInferencePipeline.inference``:
      - i2v / non-independent-first-frame: ``num_frames % num_frame_per_block == 0``
      - independent first frame (t2v):     ``(num_frames - 1) % num_frame_per_block == 0``
    """
    # A duration is a time span, so both endpoints count: 60 s at 16 fps is
    # frames 0..960 (961 frames), not 960 frames spanning only 59.9375 s.
    total_output_frames = int(round(total_duration_seconds * fps)) + 1

    if has_initial_latent:
        # i2v: the first frame is supplied as initial_latent (1 pixel frame).
        base_latent_frames = (total_output_frames - 1) // temporal_compression
        latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
        latent_frames = max(latent_frames, num_frame_per_block)
    elif independent_first_frame:
        # t2v with an independent first latent frame.
        base_latent_frames = (total_output_frames - 1) // temporal_compression
        if base_latent_frames == 0:
            latent_frames = 1
        else:
            k = math.ceil(base_latent_frames / num_frame_per_block)
            latent_frames = 1 + k * num_frame_per_block
    else:
        # t2v, blockwise from the start.
        base_latent_frames = total_output_frames // temporal_compression
        latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
        latent_frames = max(latent_frames, num_frame_per_block)

    return latent_frames


def resolve_num_latent_frames(prompt_text, has_initial_latent, pipeline, config, args):
    """Pick the noise latent-frame count from the prompt duration, falling back
    to ``--num_output_frames`` when the prompt has no duration marker."""
    fps = 16.0
    temporal_compression = 4
    num_frame_per_block = getattr(
        pipeline, 'num_frame_per_block', getattr(config, 'num_frame_per_block', 1))
    independent_first_frame = getattr(
        pipeline, 'independent_first_frame', getattr(config, 'independent_first_frame', False))

    total_duration = parse_durations_from_prompt(prompt_text)
    if total_duration is not None:
        num_latent_frames = calculate_latent_frames_from_duration(
            total_duration, fps, temporal_compression, num_frame_per_block,
            independent_first_frame, has_initial_latent)
        print(f"Duration-based length: {total_duration}s -> {num_latent_frames} latent frames")
        return num_latent_frames

    if args.num_output_frames is None:
        raise ValueError(
            "Prompt has no duration marker (e.g. [60s]); pass --num_output_frames.")
    num_latent_frames = args.num_output_frames
    if has_initial_latent:
        # For i2v the first frame is provided as initial_latent.
        num_latent_frames = args.num_output_frames - 1
    return num_latent_frames


def load_region_freeze_mask(path, latent_hw, device, dtype, invert=False, feather=0.0, erode=0.0):
    """Load a region mask file and return a per-pixel freeze weight, or None.

    Region-forcing (static-drift control): pull the static background toward a
    reference (anchor image or previous frame) while leaving the flow region
    (water/fire/smoke/...) free to be generated.

    Convention: WHITE (1) in the mask marks the DYNAMIC/flow region (model free);
    the static background (frozen toward the reference) is the complement. Pass
    --region_mask_invert if your mask paints the static region white instead.

    `erode` shrinks the frozen (static) region by that many latent pixels — be
    conservative so real motion near the flow boundary is not frozen. `feather`
    then softens the boundary to avoid seams.

    Returns a tensor of shape [1, 1, 1, H, W] (broadcasts over batch/frame/channel)
    holding the freeze weight in [0, 1], where 1 => fully pulled to the reference.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    H, W = latent_hw
    if not os.path.exists(path):
        print(f"[region-forcing] mask not found: {path}; region-forcing skipped for this sample.")
        return None

    # [H, W] in [0, 1]; white (1) = dynamic/flow region by convention.
    m = TF.to_tensor(Image.open(path).convert("L").resize((W, H), Image.BILINEAR))[0]
    freeze = m if invert else (1.0 - m)          # 1 => freeze toward reference
    # Erode the frozen region (min-pool == erosion) so we freeze less near edges.
    if erode and erode > 0:
        r = int(erode)
        k = 2 * r + 1
        freeze = 1.0 - F.max_pool2d((1.0 - freeze)[None, None], kernel_size=k, stride=1, padding=r)[0, 0]
    if feather and feather > 0:
        k = int(2 * math.ceil(2 * feather) + 1)  # odd kernel covering ~2 sigma
        freeze = TF.gaussian_blur(freeze[None, None], kernel_size=[k, k], sigma=[feather, feather])[0, 0]
    freeze = freeze.clamp(0, 1).to(device=device, dtype=dtype)
    print(f"[region-forcing] mask '{path}' -> freeze {100*freeze.mean().item():.1f}% of frame "
          f"(erode={erode}, feather={feather})")
    return freeze.view(1, 1, 1, H, W)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--phase1_config", type=str, default=None,
                    help="Optional Phase-1 YAML overlay merged after --config_path. Keeps all "
                         "arms on one byte-identical long-horizon base config.")
parser.add_argument("--phase1_set", action="append", default=[], metavar="KEY=VALUE",
                    help="Repeatable OmegaConf dot-list override for a tracked sweep cell, e.g. "
                         "model_kwargs.sink_size=6 or phase1.dymos.gamma=0.8.")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=None,
                    help="Number of latent frames to generate. Optional: used as a fallback only "
                         "when the prompt has no duration marker (e.g. [60s]).")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--frame_hash", action="store_true",
                    help=" (opt-in): write <video>_frames_sha256.json with SHA-256 of every uint8 RGB frame "
                         "BEFORE encoding (the exact array passed to write_video), so generated frames can be compared "
                         "independently of compression. Off by default (no extra file, unchanged behaviour).")
parser.add_argument("--video_crf", type=int, default=None,
                    help=" (opt-in): H.264 CRF for the written mp4 (e.g. 10 = near-lossless). Default None keeps "
                         "torchvision's default encode, byte-identical to every earlier run.")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--report_timing", action="store_true",
                    help="Only tested on A800, for the Causal Forcing++ latency. Not make claims for other hardware like H100. For the result on H100, refer to the reported results in the Self Forcing paper.")
# --- Region-forcing (#4): freeze static background to the i2v anchor latent ---
parser.add_argument("--region_forcing", action="store_true",
                    help="i2v only: hard-pin the static background to the input image's latent "
                         "while leaving the flow region free. Masks are auto-generated per image "
                         "via CLIPSeg (and saved for inspection/editing) if not already present.")
parser.add_argument("--region_mask", type=str, default=None,
                    help="Directory where per-image masks (white=flow) are saved/loaded, named after "
                         "each input image. Default: <output_folder>/region_masks. An existing mask "
                         "is reused (so you can hand-edit and re-run); a missing one is auto-generated.")
parser.add_argument("--region_mask_invert", action="store_true",
                    help="Invert mask polarity (use when the mask paints the STATIC region white).")
# The tunables below default to None so a `region:` block in the config file
# provides the values; an explicit CLI flag still overrides the config.
parser.add_argument("--region_feather", type=float, default=None,
                    help="Gaussian feather sigma (latent px) on the mask boundary. "
                         "[config region.feather, default 2.0]")
parser.add_argument("--region_threshold", type=float, default=None,
                    help="CLIPSeg threshold for auto flow masks (0=soft). Higher=freeze more. "
                         "[config region.threshold, default 0.4]")
parser.add_argument("--region_erode", type=float, default=None,
                    help="Erode (shrink) the frozen static region by N latent px. "
                         "[config region.erode, default 0]")
# --- Static-drift correction levers (gentler than a hard freeze) -------------
parser.add_argument("--region_strength", type=float, default=None,
                    help="Freeze strength alpha in [0,1]; <1 = soft pull (try 0.3-0.6). "
                         "[config region.strength, default 1.0]")
parser.add_argument("--region_reference", type=str, default=None, choices=["anchor", "prev"],
                    help="Static region pulled toward 'anchor' (input image) or 'prev' (previous "
                         "frame, gentler). [config region.reference, default anchor]")
parser.add_argument("--region_lowfreq", action="store_true", default=None,
                    help="Correct only the LOW-frequency (color) of the static region, keeping the "
                         "model's high-frequency texture. [config region.lowfreq]")
parser.add_argument("--region_lowfreq_sigma", type=float, default=None,
                    help="Gaussian sigma (latent px) for the --region_lowfreq low-pass. "
                         "[config region.lowfreq_sigma, default 1.5]")
parser.add_argument("--region_freeze_cache", action="store_true", default=None,
                    help="Also write the corrected latent into the KV cache (stronger but more OOD). "
                         "[config region.freeze_cache, default false]")
# --- Global color anchor (color/illumination drift only, motion-preserving) ---
parser.add_argument("--color_anchor", action="store_true", default=None,
                    help="i2v: pull the per-channel DC (global color) of every frame toward the input "
                         "image. Touches only the color band -> holds color identity without "
                         "constraining motion. [config color_anchor.enabled]")
parser.add_argument("--color_anchor_strength", type=float, default=None,
                    help="Blend toward the reference color in [0,1]; 1.0 = exact per-frame DC match. "
                         "Safe to set high. [config color_anchor.strength, default 0.5]")
parser.add_argument("--color_anchor_sigma", type=float, default=None,
                    help="0 = pure per-channel DC (mean) match; >0 = large-sigma low-pass color field "
                         "(latent px). [config color_anchor.sigma, default 0]")
parser.add_argument("--color_anchor_reference", type=str, default=None, choices=["anchor", "prev"],
                    help="Color pulled toward 'anchor' (input image — holds identity over time) or "
                         "'prev' (previous frame — only damps flicker). [config color_anchor.reference]")
parser.add_argument("--sink_size", type=int, default=None,
                    help="Override KV-cache sink_size (frames kept on eviction) for sink-size ablations. "
                         "Larger = more static anchoring; too large can stagnate motion.")
parser.add_argument("--ema_sink", type=int, default=None, choices=[0, 1],
                    help="Override ema_sink (0/1). 0 = pure crisp V-sink (all sink frames kept crisp, "
                         "no EMA averaging); 1 = dual-sink with EMA. For deeper-V-sink ablations.")
parser.add_argument("--motion_gamma", type=float, default=0.0,
                    help="DyMoS motion slider (training-free): subtract this from the attention "
                         "logits on the frame-0 (reference) keys -> more dynamic motion. 0=off "
                         "(baseline). Try 1.0-4.0. Higher = more motion (too high -> incoherent).")
parser.add_argument("--motion_gamma_region", action="store_true",
                    help="Apply --motion_gamma only in the fluid region (uses the same CLIPSeg "
                         "mask as --region_forcing): frees the flow while keeping the background "
                         "pinned to the reference frame. Requires --region_forcing/--region_mask.")
# --- Anchored-init / image-anchor (training-free, inference-only) -------------
# Run a T2V checkpoint as pure T2V (NO prefix-i2v frame, so no framewise-only
# assertion) but inject the DDPM-noised input-image latent into the denoising
# trajectory so the rollout conforms to the image while dynamics still flow.
#   --anchor_init_strength 0.0 == Baseline B (prompt-only T2V on the image set)
#   --anchor_init_strength >0   == Experiment C (soft anchored init)
# Needs an image-prompt dataset (--data_path to the i2v LMDB).
parser.add_argument("--anchor_init", action="store_true",
                    help="Inject the input image as a soft latent anchor into a T2V rollout "
                         "(training-free). Loads image-prompt pairs but generates as T2V "
                         "(initial_latent=None), so chunkwise/framewise both work.")
parser.add_argument("--anchor_init_strength", type=float, default=0.5,
                    help="Anchor pull alpha in [0,1] toward the image on the clean prediction. "
                         "0=pure T2V (Baseline B). With --anchor_mask it pulls only the static "
                         "region, so it can be high (0.5-1.0) without freezing the fluid. [default 0.5]")
parser.add_argument("--anchor_init_tmax", type=float, default=None,
                    help="If set, only anchor at steps whose timestep <= this value (anchor the "
                         "low-noise/appearance phase, leave high-noise structure free). [default off]")
parser.add_argument("--anchor_mask", type=str, default=None,
                    help="anchor_init: directory of per-image masks (white=fluid/flow). REGION-GATES "
                         "the anchor so only the static background is pulled to the image; the fluid "
                         "region stays free (kills the global-anchor oscillation). Missing masks are "
                         "auto-generated via CLIPSeg; --region_feather/erode/threshold/_invert apply.")
parser.add_argument("--anchor_skip_last", action="store_true",
                    help="anchor_init: do NOT correct the final denoising step (purer TTC "
                         "assimilation; gentler, less hard-paste).")
# --- Native i2v (CLIP + concat, in_dim 36): the minted i2v student ------------
parser.add_argument("--native_i2v", action="store_true",
                    help="Native Wan-I2V conditioning (model_type='i2v', in_dim 36): condition on "
                         "CLIP image features + the y-concat latent, NOT the prefix-frame KV trick. "
                         "Requires an i2v-architecture generator (e.g. the minted 1.3B-i2v seed).")
parser.add_argument("--anchor_period", type=int, default=0,
                    help="Native-i2v persistent-anchor probe: re-inject the image latent into y at "
                         "frames 0,P,2P,... to fight long-rollout drift. 0=frame-0-only (Stage-1 default); "
                         "6=re-anchor each block; 1=every frame. OOD vs training — diagnostic A/B.")
parser.add_argument("--force_num_frames", type=int, default=None,
                    help="Override the generated/noise latent-frame count, ignoring any [Ns] "
                         "duration marker. Useful only for bounded smoke tests. For nfpb=6 choose "
                         "a multiple of 6 in prefix-I2V, or 1+6k in independent-first-frame T2V.")
args = parser.parse_args()

if args.region_forcing:
    assert args.i2v, "--region_forcing is i2v-only (it needs the input image as the static anchor)."
if args.color_anchor:
    assert args.i2v, "--color_anchor is i2v-only (it needs the input image as the color reference)."
if args.anchor_init:
    assert not args.i2v, ("--anchor_init runs as T2V with a soft image anchor; it does NOT use the "
                          "prefix-i2v path. Do not combine with --i2v.")
start_time = time.time()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

# --- Partition-intervention support (opt-in, default behaviour unchanged) ------
# CausalInferencePipeline calls torch.randn_like() once per block inside the
# denoising loop, so the GLOBAL RNG state advances by an amount that depends on
# num_frame_per_block. Verified on CUDA with the real block counts (239 / 80 / 40):
# clip 0 receives identical noise across partitions, every LATER clip does not --
# and nfpb=3 vs 6 desync even though they draw equal element totals, because the
# CUDA Philox offset advances per call, not purely per element.
#
# Any experiment whose only intended factor is the temporal partition must
# therefore re-seed immediately before each clip's noise draw, so every arm sees
# the same absolute-time-indexed noise field. Off by default: enabling it changes
# the noise realisation and would break comparability with already-generated runs.
PER_PROMPT_SEED = os.environ.get("SNF_PER_PROMPT_SEED", "0") == "1"


def _reseed_for_prompt(idx):
    """Reset RNG to seed+idx so the noise field is partition-invariant."""
    if PER_PROMPT_SEED:
        set_seed(args.seed + idx)


print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
if args.phase1_config:
    config = OmegaConf.merge(config, OmegaConf.load(args.phase1_config))
if args.phase1_set:
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.phase1_set))

# Resolve region-forcing tunables: explicit CLI flag > config `region:` block >
# hard default. (The on-switches --region_forcing / --region_mask stay on the CLI.)
_region_cfg = config.get("region", {}) or {}
def _resolve_region(cli_val, key, default):
    return cli_val if cli_val is not None else _region_cfg.get(key, default)
args.region_feather = _resolve_region(args.region_feather, "feather", 2.0)
args.region_threshold = _resolve_region(args.region_threshold, "threshold", 0.4)
args.region_erode = _resolve_region(args.region_erode, "erode", 0.0)
args.region_strength = _resolve_region(args.region_strength, "strength", 1.0)
args.region_reference = _resolve_region(args.region_reference, "reference", "anchor")
args.region_lowfreq = bool(_resolve_region(args.region_lowfreq, "lowfreq", False))
args.region_lowfreq_sigma = _resolve_region(args.region_lowfreq_sigma, "lowfreq_sigma", 1.5)
args.region_freeze_cache = bool(_resolve_region(args.region_freeze_cache, "freeze_cache", False))
if args.region_forcing:
    print(f"[region-forcing] strength={args.region_strength} reference={args.region_reference} "
          f"lowfreq={args.region_lowfreq}(sigma={args.region_lowfreq_sigma}) "
          f"erode={args.region_erode} feather={args.region_feather} "
          f"freeze_cache={args.region_freeze_cache}")

# Resolve color-anchor tunables (config `color_anchor:` block, CLI overrides).
_color_cfg_yaml = config.get("color_anchor", {}) or {}
def _resolve_color(cli_val, key, default):
    return cli_val if cli_val is not None else _color_cfg_yaml.get(key, default)
args.color_anchor = bool(_resolve_color(args.color_anchor, "enabled", False))
args.color_anchor_strength = _resolve_color(args.color_anchor_strength, "strength", 0.5)
args.color_anchor_sigma = _resolve_color(args.color_anchor_sigma, "sigma", 0.0)
args.color_anchor_reference = _resolve_color(args.color_anchor_reference, "reference", "anchor")
if args.color_anchor:
    print(f"[color-anchor] strength={args.color_anchor_strength} "
          f"sigma={args.color_anchor_sigma} reference={args.color_anchor_reference}")

# C3 circularity guard: the online role map must never coexist with any mask CLI flag
# (region / anchor / region-gated DyMoS masks are evaluation-side knowledge).
if (getattr(config, "factorial", None) or {}).get("role_routing"):
    _mask_flags = {k: getattr(args, k) for k in ("region_forcing", "region_mask", "anchor_mask",
                                                  "motion_gamma_region", "anchor_init")
                   if getattr(args, k, None)}
    if _mask_flags:
        raise SystemExit(f"factorial.role_routing is model-intrinsic: refuse mask flags {_mask_flags}")

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    # Pick the generator weights robustly across checkpoint formats:
    #   - our trained ckpts: {'generator'} or {'generator_ema'}
    #   - raw seeds (e.g. rewardforcing.pt): a bare {'model': ...} state dict
    #   - or an already-unwrapped state dict (keys start with 'model.')
    if isinstance(state_dict, dict) and ('generator' in state_dict or 'generator_ema' in state_dict):
        pref = 'generator_ema' if args.use_ema else 'generator'
        if pref not in state_dict:                       # fall back to whichever exists
            pref = 'generator' if 'generator' in state_dict else 'generator_ema'
            print(f"[load] '{('generator_ema' if args.use_ema else 'generator')}' not in ckpt; using '{pref}'")
        gen_sd = state_dict[pref]
    elif isinstance(state_dict, dict) and 'model' in state_dict and isinstance(state_dict['model'], dict):
        print("[load] no 'generator' key; using bare 'model' state dict (raw seed, e.g. rewardforcing.pt)")
        gen_sd = state_dict['model']
    else:
        print("[load] using checkpoint as a raw generator state dict")
        gen_sd = state_dict

    fixed = {}
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v
    missing, unexpected = pipeline.generator.load_state_dict(fixed, strict=False)
    print(f"[load] matched {len(fixed) - len(unexpected)}/{len(fixed)} keys "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

pipeline = pipeline.to(dtype=torch.bfloat16)

# Optional sink-size override (for ablations): sink_size is the number of frames
# kept during KV-cache eviction (frame-0 hard anchor + EMA sink). It's a plain
# attribute, not a weight, so it can be set after loading. Larger = more static
# anchoring (and, past a point, more motion stagnation).
if getattr(args, "sink_size", None) is not None or getattr(args, "ema_sink", None) is not None:
    _n = 0
    for _blk in pipeline.generator.model.blocks:
        if hasattr(_blk, "self_attn"):
            if args.sink_size is not None:
                _blk.self_attn.sink_size = args.sink_size
            if args.ema_sink is not None:
                _blk.self_attn.ema_sink = bool(args.ema_sink)
            _n += 1
    print(f"[sink override] sink_size={args.sink_size} ema_sink={args.ema_sink} on {_n} blocks")

# DyMoS motion slider (training-free): subtract motion_gamma from frame-0 key logits.
if getattr(args, "motion_gamma", 0.0) and args.motion_gamma > 0.0:
    from wan21.modules import causal_model as _cm
    _cm.set_motion_gamma(args.motion_gamma)   # whole-frame; fluid-only wiring TODO
    print(f"[DyMoS] motion_gamma={args.motion_gamma} (whole-frame)")
    if args.motion_gamma_region:
        print("[DyMoS][note] --motion_gamma_region: the region-gated engine is in place "
              "(causal_model._MOTION_FLUID) but the per-image CLIPSeg->query-weight hook "
              "is not wired yet; applying whole-frame gamma for now.")

if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# UnStep C6 (factorial.svd_vo): truncated SVD of self-attention V / O after the checkpoint is
# loaded (bf16 weights -> float32 SVD -> bf16, as in tools/baselines/unstep); ranks to the output folder and RUN.json
if getattr(pipeline, "svd_vo", None) is not None:
    _svd_rows = pipeline.apply_svd_vo()
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "svd_vo.json"), "w") as _f:
        json.dump(dict(spec=pipeline.svd_vo["spec"], ranks=_svd_rows), _f, indent=1)
    _rj = os.path.join(args.output_folder, "RUN.json")
    if os.path.exists(_rj):
        _r = json.load(open(_rj))
        _r["svd_vo_ranks"] = _svd_rows            # non-contract key (cell.py compares CONTRACT keys only)
        json.dump(_r, open(_rj + ".svd.tmp", "w"), indent=1)
        os.replace(_rj + ".svd.tmp", _rj)


# Native-i2v CLIP image encoder (loaded once; weights ship in the I2V-14B dir).
clip_model = None
if args.native_i2v:
    from utils.i2v_conditioning import load_clip, build_i2v_conditioning
    clip_model = load_clip(device)

# Create dataset
if args.i2v or args.anchor_init or args.native_i2v:
    assert not dist.is_initialized(), "I2V / anchor_init does not support distributed inference yet"
    # native-i2v must feed the SAME latent shape the student trained on. The teacher+canvas
    # corpus is 58x104 latent <- 464x832 px (VAE is exact 8x), so resize inputs to 464x832.
    # (plain --i2v / --anchor_init keep the legacy 480x832.)
    in_h, in_w = (464, 832) if args.native_i2v else (480, 832)
    transform = transforms.Compose([
        transforms.Resize((in_h, in_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    print(f"[infer] input resize -> {in_h}x{in_w} px (latent {in_h//8}x{in_w//8})")
    dataset = TextImagePairDataset(
        args.data_path, transform=transform,
        eval_first_n=int(getattr(config, "eval_first_n", -1)))
else:
    dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if args.report_timing and num_prompts < 2:
    print(f"[WARN] --report_timing requires at least 2 prompts "
          f"(got {num_prompts}); timing disabled.")
    args.report_timing = False

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


_ctx_videos = []            # per-video context accounting (factorial.context_accounting)
_guidance_videos = []       # per-video prompt-extrapolation records
_forget_refs = []           # per-video redundancy reference (srcred controls)
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    _reseed_for_prompt(i)   # no-op unless SNF_PER_PROMPT_SEED=1
    if getattr(pipeline, "latent_indexed_renoise", False):
        # factorial.latent_indexed_renoise: per-(latent, timestep) re-noise generators,
        # keyed like the per-prompt seed so every granularity sees the same field.
        pipeline.renoise_seed = args.seed + i
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    region_freeze = None  # region-forcing mask (i2v only); None disables it
    anchor_cfg = None     # anchored-init image anchor (anchor_init only); None disables it
    clip_fea_cond = None  # native-i2v CLIP features; None disables native-i2v conditioning
    y_cond = None         # native-i2v conditioning latent (per-frame y); None for t2v


    if args.native_i2v:
        # Native Wan-I2V: condition on CLIP features + the y-concat latent (in_dim 36),
        # generated as a T2V-style rollout (initial_latent=None — the image rides in
        # clip_fea/y, not a prefix KV frame). y is sliced per block inside the pipeline.
        raw_prompt = batch['prompts'][0]
        prompt = strip_duration_markers(raw_prompt)
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        image_pix = batch['image'][0].to(device=device, dtype=torch.float32)  # [C,H,W] in [-1,1]
        prompts = [prompt]
        initial_latent = None
        num_latent_frames = args.force_num_frames if args.force_num_frames else resolve_num_latent_frames(
            raw_prompt, has_initial_latent=False, pipeline=pipeline, config=config, args=args)
        clip_fea_cond, y_cond = build_i2v_conditioning(
            image_pix, pipeline.vae, clip_model, num_latent_frames, anchor_period=args.anchor_period)
        clip_fea_cond = clip_fea_cond.to(dtype=torch.bfloat16)
        y_cond = y_cond.to(dtype=torch.bfloat16)
        print(f"[native-i2v] clip_fea={tuple(clip_fea_cond.shape)} y={tuple(y_cond.shape)} "
              f"frames={num_latent_frames}")
        # noise MUST match the conditioning latent's spatial shape (derived from the input
        # image, e.g. 58x104), not a hardcoded 60x104 — else the in_dim=36 concat mismatches.
        lh, lw = y_cond.shape[-2], y_cond.shape[-1]
        sampled_noise = torch.randn(
            [1, num_latent_frames, 16, lh, lw], device=device, dtype=torch.bfloat16)
    elif args.anchor_init:
        # Anchored-init: load the image-prompt pair, but generate as pure T2V
        # (initial_latent=None) and feed the image as a soft trajectory anchor.
        raw_prompt = batch['prompts'][0]
        prompt = strip_duration_markers(raw_prompt)
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        image_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt]
        initial_latent = None  # NOT prefix-i2v: run as plain T2V so the anchor is the only image signal

        # Optional region gate: build a static-anchor weight (1=static, 0=fluid) so the
        # anchor only holds the background and the fluid region stays free. Reuses the
        # CLIPSeg flow-mask machinery (white=flow) -> load_region_freeze_mask returns the
        # static-side weight, which is exactly the per-pixel anchor strength we want.
        anchor_mask = None
        if args.anchor_mask:
            import torchvision.transforms.functional as TF
            os.makedirs(args.anchor_mask, exist_ok=True)
            stem = os.path.splitext(str(batch['file_name'][0]))[0] if 'file_name' in batch else f"sample_{idx}"
            mask_path = os.path.join(args.anchor_mask, f"{stem}.png")
            if not os.path.exists(mask_path):
                img01 = (batch['image'][0].float() * 0.5 + 0.5).clamp(0, 1)
                pil_img = TF.to_pil_image(img01)
                queries = extract_flow_queries(prompt)
                flow_mask = generate_flow_mask_clipseg(
                    pil_img, queries, device=device, threshold=args.region_threshold)
                flow_mask.save(mask_path)
                print(f"[anchor-mask] auto-generated for '{stem}' (flow queries: {queries}) -> {mask_path}")
            anchor_mask = load_region_freeze_mask(
                mask_path, latent_hw=(image_latent.shape[-2], image_latent.shape[-1]),
                device=device, dtype=image_latent.dtype,
                invert=args.region_mask_invert, feather=args.region_feather, erode=args.region_erode)

        anchor_cfg = {
            "enabled": args.anchor_init_strength > 0.0,
            "latent": image_latent[:, :1],
            "strength": args.anchor_init_strength,
            "mask": anchor_mask,            # 1=static/anchor, 0=fluid/free; None=global
            "tmax": args.anchor_init_tmax,
            "skip_last": args.anchor_skip_last,
        }
        print(f"[anchor-init] strength={args.anchor_init_strength} "
              f"gate={'region' if anchor_mask is not None else 'global'} "
              f"tmax={args.anchor_init_tmax} skip_last={args.anchor_skip_last} "
              f"(strength=0 -> Baseline B prompt-only)")
        num_latent_frames = resolve_num_latent_frames(
            raw_prompt, has_initial_latent=False, pipeline=pipeline, config=config, args=args)
        sampled_noise = torch.randn(
            [1, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    elif args.i2v:
        # Framewise i2v (num_frame_per_block == 1) needs no special handling.
        # Chunkwise i2v (num_frame_per_block > 1) is only valid with
        # independent_first_frame: true, so the image becomes a standalone anchor
        # frame followed by [N, N, ...] chunks (the [1, N, N, ...] structure).
        if config.num_frame_per_block > 1:
            assert getattr(config, "independent_first_frame", False), (
                "Chunkwise i2v (num_frame_per_block > 1) requires independent_first_frame: true "
                "in the config so the image is an independent anchor frame followed by chunks."
            )
        # For image-to-video, batch contains image and caption
        raw_prompt = batch['prompts'][0]  # Get caption from batch (may carry a [Ns] marker)
        prompt = strip_duration_markers(raw_prompt)  # what we actually encode
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt]
        _fgc = getattr(pipeline, "forget", None)
        if _fgc is not None and _fgc.get("reference") == "xref":
            # fg_srcred_xref: the redundancy reference is the source of ANOTHER scene of this split, by a
            # fixed cyclic derangement (+1 over the dataset order; a derangement for >= 2 scenes); its image is encoded
            # exactly like this one and only used for a scratch prefix pass inside the pipeline (never conditioning)
            if len(dataset) < 2:
                raise SystemExit("forget reference xref needs >= 2 scenes in the split")
            _j = (int(idx) + 1) % len(dataset)
            _other = dataset[_j]
            _oimg = _other['image'].unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            pipeline.forget_xref_latent = pipeline.vae.encode_to_latent(_oimg).to(device=device, dtype=torch.bfloat16)
            pipeline.forget_xref_scene = str(_other['file_name'])
            print(f"[forget] xref: scene '{batch['file_name'][0]}' uses the source of '{_other['file_name']}' "
                  f"(index {idx} -> {_j}, cyclic +1 derangement) as its redundancy reference")
        # Region-forcing (#4): build the static-background freeze mask for this sample.
        if args.region_forcing:
            import torchvision.transforms.functional as TF
            # Masks live in a directory and are named after the input image, so
            # they are easy to trace back to their prompt/image and hand-edit.
            region_dir = args.region_mask or os.path.join(args.output_folder, "region_masks")
            os.makedirs(region_dir, exist_ok=True)
            stem = os.path.splitext(str(batch['file_name'][0]))[0] if 'file_name' in batch else f"sample_{idx}"
            mask_path = os.path.join(region_dir, f"{stem}.png")
            # Auto-generate (CLIPSeg) only if a mask isn't already there; an
            # existing mask is reused so manual edits survive re-runs.
            if not os.path.exists(mask_path):
                img01 = (batch['image'][0].float() * 0.5 + 0.5).clamp(0, 1)  # de-normalize -> [0,1] RGB
                pil_img = TF.to_pil_image(img01)
                queries = extract_flow_queries(prompt)
                flow_mask = generate_flow_mask_clipseg(
                    pil_img, queries, device=device, threshold=args.region_threshold)
                flow_mask.save(mask_path)
                print(f"[region-forcing] auto-generated mask for '{stem}' "
                      f"(flow queries: {queries}) -> {mask_path}")
            region_freeze = load_region_freeze_mask(
                mask_path,
                latent_hw=(initial_latent.shape[-2], initial_latent.shape[-1]),
                device=device, dtype=initial_latent.dtype,
                invert=args.region_mask_invert, feather=args.region_feather,
                erode=args.region_erode)
        num_latent_frames = args.force_num_frames if args.force_num_frames else resolve_num_latent_frames(
            raw_prompt, has_initial_latent=True, pipeline=pipeline, config=config, args=args)
        sampled_noise = torch.randn(
            [1, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        raw_prompt = batch['prompts'][0]
        prompt = strip_duration_markers(raw_prompt)
        output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            raw_for_duration = extended_prompt
            prompts = [strip_duration_markers(extended_prompt)]
        else:
            raw_for_duration = raw_prompt
            prompts = [prompt]

        initial_latent = None
        num_latent_frames = resolve_num_latent_frames(
            raw_for_duration, has_initial_latent=False, pipeline=pipeline, config=config, args=args)
        sampled_noise = torch.randn(
            [1, num_latent_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    _g = getattr(pipeline, "guidance", None)
    if _g is not None and _g["branch"] == "counter":
        #: the matched counter-prompt of THIS scene, by image file name; the stored base text must be
        # exactly the caption being encoded (the exact pair is recorded in the context json and RUN.json)
        _fn = str(batch['file_name'][0]) if 'file_name' in batch else None
        _pair = _g["counter"]["by_file"].get(_fn)
        if _pair is None:
            raise SystemExit(f"guidance counter: no counter prompt for '{_fn}' in {_g['counter']['path']}")
        if _pair["base_text"] != prompts[0]:
            raise SystemExit(f"guidance counter: stored base text for '{_fn}' differs from the encoded caption")
        pipeline.guidance_alt_text = _pair["counter_text"]

    inference_start_time = time.time()
    sample_report_timing = args.report_timing and i >= 1
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        report_timing=sample_report_timing,
        region_freeze=region_freeze,
        region_cfg={
            "strength": args.region_strength,
            "reference": args.region_reference,
            "lowfreq": args.region_lowfreq,
            "lowfreq_sigma": args.region_lowfreq_sigma,
            "write_cache": args.region_freeze_cache,
        },
        color_cfg={
            "enabled": args.color_anchor,
            "strength": args.color_anchor_strength,
            "sigma": args.color_anchor_sigma,
            "reference": args.color_anchor_reference,
        },
        anchor_cfg=anchor_cfg,
        clip_fea=clip_fea_cond,
        y=y_cond,
    )

    # Go/no-go metrics: motion energy (mean frame-to-frame latent change) and, when
    # an image anchor is present, color/identity drift from the anchor. Lets you read
    # the motion-vs-identity trade-off straight from the log without opening the mp4.
    with torch.no_grad():
        _lat = latents.float()
        motion_e = (_lat[:, 1:] - _lat[:, :-1]).abs().mean().item() if _lat.shape[1] > 1 else 0.0
        _msg = f"[metrics] motion_energy(|Δframe|)={motion_e:.4f}"
        if anchor_cfg is not None and anchor_cfg.get("latent") is not None:
            _a = anchor_cfg["latent"].float()
            _drift = (_lat.mean(dim=(-1, -2)) - _a.mean(dim=(-1, -2))).abs().mean().item()
            _msg += f"  anchor_color_drift(|μframe-μanchor|)={_drift:.4f}"
        print(_msg)
    if sample_report_timing:
        latency = pipeline.first_chunk_time
        elapsed = pipeline.last_generation_time
        num_pixel_frames = video.shape[1]
        fps = num_pixel_frames / elapsed if elapsed > 0 else float('inf')
        print(f"[Sample {i}] {num_pixel_frames} frames, "
              f"latency ↓ {latency:.2f}s, FPS ↑ {fps:.2f}")
        # Only tested on A800, for the Causal Forcing++ paper latency & throughput.
        # Not make claims for other hardware like H100.
        # For the result on H100, refer to the reported results in the Self Forcing paper.
        # We do not guarantee that our FPS/latency measurement protocol is identical to that used in the Self Forcing paper.
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    output_path = os.path.join(args.output_folder, f'{prompt[:100]}.mp4')
    if args.frame_hash:
        import hashlib as _hl, json as _js
        _u8 = torch.as_tensor(video[0], dtype=torch.uint8).cpu().contiguous()   # the cast write_video applies
        _per = [_hl.sha256(_u8[_t].numpy().tobytes()).hexdigest() for _t in range(_u8.shape[0])]
        with open(output_path[:-4] + "_frames_sha256.json", "w") as _f:
            _js.dump(dict(frames=int(_u8.shape[0]), shape=list(_u8.shape), dtype="uint8 RGB, pre-encode",
                          all_sha256=_hl.sha256(_u8.numpy().tobytes()).hexdigest(), per_frame_sha256=_per), _f)
    if args.video_crf is None:
        write_video(output_path, video[0], fps=16)
    else:
        write_video(output_path, video[0], fps=16, options={"crf": str(int(args.video_crf))})
    print(f"Saved video to {output_path} (generated {num_generated_frames} latent frames); single-sample elapsed time: {time.time() - inference_start_time:.2f}s")
    _rmap = getattr(pipeline, "role_routing_map", None)
    if _rmap is not None:
        # C3: the frozen online map of this video (token grid; white = low temporal variation
        # (`support`, sharp entry every step), black = high temporal variation (`material`, scheduled
        # entry), grey = uncertain band), for an AFTER-THE-FACT agreement report against the
        # evaluation masks (never read by generation). Upscaled x8 nearest so a 30x52 grid is viewable.
        # `shuffled: true` marks the spatially permuted control (the real map is kept as *_unshuffled).
        from PIL import Image as _Image
        _lab = _rmap["labels"].numpy()
        _pix = (255 * (_lab == 0) + 128 * (_lab == 2)).astype("uint8")
        _img = _Image.fromarray(_pix, mode="L")
        _img = _img.resize((_img.width * 8, _img.height * 8), resample=_Image.NEAREST)
        _rp = os.path.join(args.output_folder, f'{prompt[:100]}_routing.png')
        _img.save(_rp)
        _grid_keys = ("support", "log_var", "labels", "support_unshuffled", "labels_unshuffled")
        with open(os.path.join(args.output_folder, f'{prompt[:100]}_routing.json'), "w") as _f:
            json.dump({k: v for k, v in _rmap.items() if k not in _grid_keys}
                      | {"token_grid": list(_rmap["support"].shape), "white": "support (low temporal variation)",
                         "black": "material (high temporal variation)", "grey": "uncertain band",
                         "log10_var": _rmap["log_var"].tolist(), "labels": _rmap["labels"].tolist()}
                      | ({"labels_unshuffled": _rmap["labels_unshuffled"].tolist()} if _rmap.get("shuffled") else {}),
                      _f)
        print(f"[role_routing] wrote {_rp} (low-variation {100 * _rmap['support_fraction']:.1f} %, "
              f"high-variation {100 * _rmap['material_fraction']:.1f} %, uncertain {100 * _rmap['uncertain_fraction']:.1f} %, "
              f"n_latents {_rmap['n_latents']}{', SHUFFLED control' if _rmap.get('shuffled') else ''})")
    _fev = getattr(pipeline, "forget_events", None) if getattr(pipeline, "forget", None) is not None else None
    if _fev:
        #: the online role maps of this video (the last one as png: white = rigid / low temporal
        # variation, black = material / high; every refresh in the json), for an AFTER-THE-FACT comparison with the
        # evaluation masks (never read by generation)
        from PIL import Image as _Image
        _last = _fev[-1]["material"].numpy()
        _img = _Image.fromarray(((~_last).astype("uint8") * 255), mode="L")
        _img = _img.resize((_img.width * 8, _img.height * 8), resample=_Image.NEAREST)
        _img.save(os.path.join(args.output_folder, f'{prompt[:100]}_forget_map.png'))
        with open(os.path.join(args.output_folder, f'{prompt[:100]}_forget_map.json'), "w") as _f:
            json.dump(dict(config={k: v for k, v in pipeline.forget.items()}, white="rigid (low temporal variation)",
                           black="material (high temporal variation)",
                           events=[{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in e.items()} for e in _fev]), _f)
        print(f"[forget] wrote {len(_fev)} map(s) for this video; refresh times "
              + ", ".join(f"{e['t_s']:.2f}s" for e in _fev) + f"; material fractions "
              + ", ".join(f"{100 * e['material_fraction']:.0f}%" for e in _fev))
    if getattr(pipeline, "context_accounting", False):
        #: per-video context json (schedule record + static KV allocation + one row per block)
        _rows = list(pipeline.context_rows)
        _cj = dict(schedule=pipeline.schedule_record, static=pipeline.context_static_info, blocks=_rows,
                   video=f'{prompt[:100]}.mp4')
        _fgc2 = getattr(pipeline, "forget", None)
        if _fgc2 is not None and _fgc2.get("roles") == "srcred":
            #: which reference the redundancy score used for this video
            _cj["forget_reference"] = dict(reference=_fgc2.get("reference", "source"), redundancy=_fgc2.get("redundancy"),
                                           xref_scene=getattr(pipeline, "forget_xref_scene", None)
                                           if _fgc2.get("reference") == "xref" else None,
                                           derangement="cyclic +1 over the dataset order" if _fgc2.get("reference") == "xref" else None)
            _forget_refs.append(dict(video=_cj["video"], **_cj["forget_reference"]))
        _grec = getattr(pipeline, "guidance_record", None)
        if _grec is not None:
            #: exact text pair, scale, cost and per-evaluation ||z_base - z_alt|| (RMS) of this video
            _ev = _grec.get("evals", [])
            _rel = [e["rms_diff"] / max(e["rms_base"], 1e-12) for e in _ev]
            _by_t = {}
            for e, r in zip(_ev, _rel):
                _by_t.setdefault(str(e["t_model"]), []).append(r)
            _cj["guidance"] = dict(_grec, summary=dict(
                evals=len(_ev), rel_diff_mean=(sum(_rel) / len(_rel)) if _rel else None,
                rel_diff_max=max(_rel) if _rel else None,
                rel_diff_mean_by_t_model={t: sum(v) / len(v) for t, v in _by_t.items()}))
            _guidance_videos.append(dict(video=_cj["video"], branch=_grec["branch"], scale=_grec["scale"],
                                         base_text=_grec.get("base_text"), alt_text=_grec.get("alt_text"),
                                         cost=_grec.get("cost"), **_cj["guidance"]["summary"]))
        with open(os.path.join(args.output_folder, f'{prompt[:100]}_context.json'), "w") as _f:
            json.dump(_cj, _f)

        def _mean(key, rows):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None
        _late = [r for r in _rows if r["block_start"] >= 160]            # >= 40 s at 4 latents / s
        _ctx_videos.append(dict(video=_cj["video"], blocks=len(_rows),
                                **{f"{k}_mean": _mean(k, _rows) for k in ("frames", "duplicates", "recent_coverage", "blended",
                                                                         "mass_support", "mass_material", "mass_all")},
                                source_attended_frac=_mean("source_attended", _rows),
                                **{f"late_{k}_mean": _mean(k, _late) for k in ("frames", "duplicates", "recent_coverage",
                                                                               "mass_support", "mass_material", "mass_all")},
                                late_source_attended_frac=_mean("source_attended", _late)))
        print(f"[context] {len(_rows)} blocks: frames {_mean('frames', _rows):.2f} dup {_mean('duplicates', _rows):.2f} "
              f"recent {_mean('recent_coverage', _rows):.2f} source {_mean('source_attended', _rows):.2f}")

if _ctx_videos:
    _sum = dict(videos=len(_ctx_videos), static=getattr(pipeline, "context_static_info", None),
                schedule=getattr(pipeline, "schedule_record", None), per_video=_ctx_videos)
    for k in [k for k in _ctx_videos[0] if k.endswith("_mean") or k.endswith("_frac")]:
        vals = [v[k] for v in _ctx_videos if v.get(k) is not None]
        _sum[k] = (sum(vals) / len(vals)) if vals else None
    with open(os.path.join(args.output_folder, "context_summary.json"), "w") as _f:
        json.dump(_sum, _f, indent=1)
    _rj = os.path.join(args.output_folder, "RUN.json")
    if os.path.exists(_rj):
        # add the summary to RUN.json (a non-contract key; tools/factorial/cell.py compares CONTRACT keys only)
        _r = json.load(open(_rj))
        _r["context_summary"] = {k: v for k, v in _sum.items() if k != "per_video"}
        if _forget_refs:
            _r["forget_reference"] = _forget_refs      # (non-contract key)
        if _guidance_videos:
            #: texts (the exact base / alternative pair per video), scale, branch and cost
            _gcfg = pipeline.guidance
            _r["guidance"] = dict(mode=_gcfg["mode"], scale=_gcfg["scale"], branch=_gcfg["branch"],
                                  fixed_alt_text=_gcfg.get("text"),
                                  counter_prompts=(dict(path=_gcfg["counter"]["path"], sha256=_gcfg["counter"]["sha256"])
                                                   if _gcfg.get("counter") else None),
                                  per_video=_guidance_videos,
                                  note="prompt extrapolation z = z_base + (s-1)(z_base - z_alt) on a DMD-distilled student "
                                       "(guidance baked in), not CFG; 2x DiT forwards on denoising evaluations")
        _tmp = _rj + ".ctx.tmp"
        json.dump(_r, open(_tmp, "w"), indent=1)
        os.replace(_tmp, _rj)
        print(f"[context] summary written to {_rj} and context_summary.json")

print(f"Total elapsed time: {time.time() - start_time:.2f}s")
