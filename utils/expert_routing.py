"""Timestep-routed expert selection with cross-rank agreement.

The Wan2.2 A14B teacher is two FSDP-wrapped experts with identical parameter shapes.
Every FSDP unit's forward issues an all-gather of its shard, and NCCL/gloo match
collectives by issue ORDER, not by unit identity. If rank 0 routes a step to the
high-noise expert while rank 1 routes it to the low-noise expert, rank 0's all-gather of
`high.blocks[i]` is answered with rank 1's shard of `low.blocks[i]`: both ranks assemble
a mixed expert and score with it, silently. The DMD timestep is sampled per rank
(model/dmd.py, seeds are `seed + rank`), so without synchronization this happens on
every step whose per-rank draws straddle the boundary.

Two guards, both cheap and both usable from CPU tests (this module imports nothing
from `wan21`, whose package __init__ needs a CUDA device):

  broadcast_timestep   make the sampled timestep identical on all ranks (fix)
  assert_expert_sync   all_gather the chosen expert id and fail loudly if ranks
                       disagree (detector; runs before any expert forward)

The assertion default is "on whenever torch.distributed is initialized" and can be
forced either way with the environment variable CFPP_ASSERT_EXPERT_SYNC=0/1 or the
`enabled` argument.
"""
import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist

ASSERT_EXPERT_SYNC_ENV = "CFPP_ASSERT_EXPERT_SYNC"
HIGH, LOW = "high_noise", "low_noise"


def _dist_active(group=None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def select_expert_name(t, boundary_step: float) -> str:
    """Route one batch on its timestep: t >= boundary_step -> 'high_noise', else 'low_noise'.

    This is the official Wan2.2 rule (`t >= boundary * num_train_timesteps`). A batch
    straddling the boundary would need to be split across both experts; cfpp trains at
    batch_size 1 with one timestep per batch, so a straddling batch is an error.
    """
    tv = t.detach().float().flatten() if torch.is_tensor(t) else torch.tensor([float(t)])
    hi = tv >= float(boundary_step)
    if bool(hi.all()):
        return HIGH
    if bool((~hi).all()):
        return LOW
    raise RuntimeError(
        f"MoE batch straddles the expert boundary {float(boundary_step):.1f}: "
        f"timesteps {tv.tolist()}. Split the batch by expert before calling, or "
        f"sample one timestep per batch."
    )


def assert_sync_enabled(enabled: Optional[bool] = None, group=None) -> bool:
    """Resolve the assertion flag: explicit arg > env var > (dist initialized)."""
    if enabled is not None:
        return bool(enabled)
    env = os.environ.get(ASSERT_EXPERT_SYNC_ENV)
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    return _dist_active(group)


def gather_expert_names(name: str, group=None) -> Sequence[str]:
    """all_gather the expert id chosen on this rank. Collective: every rank must call it."""
    if not _dist_active(group):
        return [name]
    names = [None] * dist.get_world_size(group)
    dist.all_gather_object(names, name, group=group)
    return names


def assert_expert_sync(name: str, enabled: Optional[bool] = None, group=None,
                       timestep=None) -> Sequence[str]:
    """Raise RuntimeError on every rank if the ranks did not all choose `name`.

    Returns the per-rank list of names (length 1 when not distributed / disabled). The
    all_gather happens BEFORE any expert forward, so a disagreement surfaces as a clear
    error rather than as a mixed-expert score.
    """
    if not assert_sync_enabled(enabled, group):
        return [name]
    names = gather_expert_names(name, group)
    if any(n != names[0] for n in names):
        rank = dist.get_rank(group) if _dist_active(group) else 0
        ts = ""
        if timestep is not None:
            tv = timestep.detach().float().flatten().tolist() if torch.is_tensor(timestep) else [float(timestep)]
            ts = f" (this rank's timestep {tv})"
        raise RuntimeError(
            "MoE expert routing disagrees across ranks: "
            + ", ".join(f"rank{i}={n}" for i, n in enumerate(names))
            + f"; this is rank {rank}{ts}. Under FSDP a per-rank expert choice assembles a "
            "MIXED expert from the other rank's shards. Broadcast the DMD "
            "timestep from rank 0 (model/dmd.py sync_dmd_timestep) so every rank routes "
            f"identically, or set {ASSERT_EXPERT_SYNC_ENV}=0 to disable this check."
        )
    return names


def broadcast_timestep(timestep: torch.Tensor, src: int = 0, enabled: bool = True,
                       group=None) -> torch.Tensor:
    """Make the sampled timestep tensor identical on all ranks (rank `src` wins).

    In place when possible; returns the tensor. No-op when not distributed or disabled.
    """
    if not enabled or not _dist_active(group):
        return timestep
    t = timestep if timestep.is_contiguous() else timestep.contiguous()
    dist.broadcast(t, src=src, group=group)
    if t is not timestep:
        timestep.copy_(t)
    return timestep
