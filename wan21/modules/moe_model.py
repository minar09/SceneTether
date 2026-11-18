"""Wan2.2 A14B two-expert (MoE) teacher for DMD / Re-DMD.

Wan2.2's A14B models split the denoising trajectory between two ~14B experts rather than
widening one: a high-noise expert for the early layout stage and a low-noise expert for
the later detail stage. Total ~27B parameters, but only 14B are active per step, so the
per-step cost matches a dense 14B model.

The switch point is defined by SNR: the official implementation
(wan22_turbo/wan22/image2video.py:189) selects the high-noise expert when
`t >= boundary * num_train_timesteps`, with `boundary = 0.900` for I2V-A14B
(wan/configs/wan_i2v_A14B.py). That exact rule is reproduced here -- it is the one thing
that must not drift, because using the wrong expert produces a plausible-looking but
systematically mis-scored teacher signal, which is the kind of error DMD would silently
train the student to imitate.

Both experts are loaded with `model_type='i2v22'`, overriding the `"i2v"` recorded in
their config.json. That override is required: Wan2.2's I2V conditioning has no CLIP branch
and no image cross-attention, so building Wan2.1's 'i2v' graph would add 208 parameter
tensors the checkpoint does not contain. See the 'i2v22' note in wan/modules/model.py.
"""
from typing import Optional

import torch
from torch import nn

from wan21.modules.model import WanModel
from utils.expert_routing import assert_expert_sync, select_expert_name


class MoEWanModel(nn.Module):
    """Two Wan2.2 experts behind one WanModel-compatible forward."""

    def __init__(
        self,
        checkpoint_dir: str,
        boundary: float = 0.900,
        num_train_timesteps: int = 1000,
        low_subfolder: str = "low_noise_model",
        high_subfolder: str = "high_noise_model",
        model_type: str = "i2v22",
        offload_inactive: bool = False,
        assert_expert_sync: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        **model_kwargs,
    ) -> None:
        super().__init__()
        # B1: None -> on whenever torch.distributed is initialized (env
        # CFPP_ASSERT_EXPERT_SYNC=0/1 overrides); True/False forces it.
        self.assert_expert_sync = assert_expert_sync
        # B2: the checkpoint is stored fp32 (57 GB per expert). `dtype` casts each
        # expert right after it is loaded (one at a time, so host RAM peaks at one fp32
        # expert + the cast copies, not two). None keeps the stored dtype. Under FSDP
        # MixedPrecision(param_dtype=bf16) the forward sees bf16 either way; casting here
        # halves the resident shards. `freqs` is a plain attribute (complex64), untouched.
        self.dtype = dtype
        self.low_noise_model = self._cast(WanModel.from_pretrained(
            checkpoint_dir, subfolder=low_subfolder, model_type=model_type, **model_kwargs))
        self.high_noise_model = self._cast(WanModel.from_pretrained(
            checkpoint_dir, subfolder=high_subfolder, model_type=model_type, **model_kwargs))

        # absolute timestep at which the experts swap (900 for I2V-A14B)
        self.boundary_step = float(boundary) * float(num_train_timesteps)
        self.offload_inactive = offload_inactive
        self.model_type = model_type
        # mirrored so callers that introspect the dense model keep working
        self.in_dim = self.low_noise_model.in_dim
        self.out_dim = self.low_noise_model.out_dim
        self.dim = self.low_noise_model.dim
        self.num_layers = self.low_noise_model.num_layers
        self.freqs = self.low_noise_model.freqs

    def _cast(self, expert: nn.Module) -> nn.Module:
        if self.dtype is not None:
            expert = expert.to(self.dtype)
        return expert

    # -- WanModel-compatible surface -------------------------------------------------
    def enable_gradient_checkpointing(self) -> None:
        self.low_noise_model.enable_gradient_checkpointing()
        self.high_noise_model.enable_gradient_checkpointing()

    def _select(self, t: torch.Tensor):
        """-> (expert, name). Routes on the timestep, refusing a mixed batch.

        A batch straddling the boundary would need to be split and run through both
        experts. cfpp trains at batch_size 1 with a uniform timestep per sample, so rather
        than carry an untested split path we fail loudly if that assumption ever breaks.

        Under FSDP every rank must pick the SAME expert: the experts have
        identical shapes, so a rank in 'high' all-gathers another rank's 'low' shards into
        a mixed expert). `assert_expert_sync` all-gathers the chosen id before any expert
        forward and raises on disagreement; it is on by default when dist is initialized.
        """
        name = select_expert_name(t, self.boundary_step)
        assert_expert_sync(name, enabled=self.assert_expert_sync, timestep=t)
        if name == "high_noise":
            return self.high_noise_model, name
        return self.low_noise_model, name

    def forward(self, x, t=None, **kwargs):
        expert, name = self._select(t)
        if self.offload_inactive:
            other = (self.low_noise_model if name == "high_noise"
                     else self.high_noise_model)
            other.to("cpu")
            expert.to(x.device if torch.is_tensor(x) else x[0].device)
        return expert(x, t=t, **kwargs)
