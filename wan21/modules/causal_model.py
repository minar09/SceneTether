import os
from wan21.modules.attention import attention
from wan21.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention,
    dynamic=False,
    mode="default"
)



def spectral_anchor_blend(local, anchor, grid_sizes, strength, sigma=0.125):
    """Current-block 3D low-pass correction, frequencies in cycles/token.

    FreqForcing-inspired, not a physical appearance/motion decomposition.
    Never transforms committed history or future blocks. Zero is exact bypass.
    """
    if strength == 0:
        return local
    if not 0 <= strength <= 1 or sigma <= 0 or local.shape != anchor.shape:
        raise ValueError('invalid spectral blend parameters/shapes')
    outputs = []
    for b, (f, h, w) in enumerate(grid_sizes.tolist()):
        if f * h * w != local.shape[1]:
            raise ValueError('spectral blend requires an unpadded current grid')
        delta = (anchor[b].float() - local[b].float()).reshape(f, h, w, -1)
        ft = torch.fft.fftfreq(f, device=delta.device)[:, None, None]
        fy = torch.fft.fftfreq(h, device=delta.device)[None, :, None]
        fx = torch.fft.fftfreq(w, device=delta.device)[None, None, :]
        lowpass = torch.exp(-(ft.square() + fy.square() + fx.square()) / (2 * sigma ** 2))
        correction = torch.fft.ifftn(torch.fft.fftn(delta, dim=(0, 1, 2)) *
                                    lowpass[..., None], dim=(0, 1, 2)).real
        outputs.append((local[b].float() + strength * correction.reshape_as(local[b])).to(local.dtype))
    return torch.stack(outputs)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _jittered_temporal_freqs(length, complex_dim, num_heads, theta_by_head,
                             start_frame, device):
    """LoL Algorithm 1 temporal frequencies for one attention layer.

    ``theta_by_head`` is sampled once when the model is built, not on every
    forward pass.  Returning [F, H, D/2] keeps spatial RoPE unchanged.
    """
    positions = torch.arange(start_frame, start_frame + length,
                             device=device, dtype=torch.float64)
    exponents = torch.arange(complex_dim, device=device, dtype=torch.float64)
    exponents = exponents.div(complex_dim)
    theta = theta_by_head.to(device=device, dtype=torch.float64).clamp_min(1.0)
    omega = theta[:, None].pow(-exponents[None, :])
    phase = positions[:, None, None] * omega[None, :, :]
    return torch.polar(torch.ones_like(phase), phase)


def block_relativistic_rope(x, grid_sizes, freqs, start_frame=0,
                            theta_by_head=None, spatial_ids=None):
    """Re-rope a token window on a *relative* temporal coordinate frame.

    Unlike ``causal_rope_apply`` (which ropes at absolute positions and is then
    cached), this is applied to *raw* keys/queries every step so the whole
    attention window slides on a relative frame. Used by the steady-forcing
    cache path together with the dual-sink + EMA cache update to keep a static,
    fixed-camera scene steady over long horizons.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers (inference precision)
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

        temporal_freqs = freqs[0][start_frame:start_frame + f]

        if spatial_ids is not None:
            # Compressed tokens retain their source-grid position. Temporal
            # placement deliberately stays packed: this is a spatial-only arm.
            ids = spatial_ids[i, :seq_len]
            if ids.dtype != torch.long or ids.numel() != seq_len:
                raise ValueError("spatial_ids must be int64 [batch, sequence]")
            hf = freqs[1][torch.div(ids, w, rounding_mode='floor')]
            wf = freqs[2][ids.remainder(w)]
            if theta_by_head is None:
                tf = temporal_freqs[:, None, :].expand(f, h * w, -1).reshape(seq_len, -1)
                freqs_i = torch.cat([tf, hf, wf], dim=-1)[:, None, :]
            else:
                tf = _jittered_temporal_freqs(
                    f, temporal_freqs.shape[-1], n, theta_by_head,
                    start_frame, x.device)
                tf = tf[:, None, :, :].expand(f, h * w, n, -1).reshape(seq_len, n, -1)
                freqs_i = torch.cat([tf, hf[:, None, :].expand(-1, n, -1),
                                     wf[:, None, :].expand(-1, n, -1)], dim=-1)
        elif theta_by_head is None:
            freqs_i = torch.cat([
                temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)
        else:
            # Temporal base differs by head; height/width bases remain shared.
            tf = _jittered_temporal_freqs(
                f, temporal_freqs.shape[-1], n, theta_by_head,
                start_frame, x.device).view(f, 1, 1, n, -1).expand(f, h, w, n, -1)
            hf = freqs[1][:h].view(1, h, 1, 1, -1).expand(f, h, w, n, -1)
            wf = freqs[2][:w].view(1, 1, w, 1, -1).expand(f, h, w, n, -1)
            freqs_i = torch.cat([tf, hf, wf], dim=-1).reshape(seq_len, n, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)

    return torch.stack(output).type_as(x)


# --- Optional attention-to-anchor capture (research viz; OFF by default) -------
# When _ATTN_CAPTURE is True (set by the trainer for a single viz step), the
# cache-path self-attention records how much the mean query attends to the frame-0
# anchor and the sink frames, per rollout frame. This is the drift diagnostic:
# anchor attention decaying as frames progress => the model is letting go of the
# input image. Mean-query approximation keeps it cheap; ZERO overhead when False.
_ATTN_CAPTURE = False
_ATTN_RECORDS = []


# --- DyMoS: Dynamic Motion Slider (inference-time, training-free; OFF by default)
# Subtract a positive scalar gamma from the attention LOGITS on the reference-frame
# (frame-0) key tokens -> reduces reference-frame dominance -> more dynamic motion.
# gamma=0 -> the fast flash `attention()` path is used unchanged (training never hits
# this). _MOTION_FLUID (optional, length = query tokens of one frame) scales the bias
# per query so only the fluid region is freed (region-gated DyMoS).
_MOTION_GAMMA = 0.0
_MOTION_FLUID = None
_MOTION_STEP_RATIO = 0.0
_DENOISE_STEP_FRACTION = None
#: True while the prompt-extrapolation alternative branch runs (its cache writes are restored
# by the pipeline); only gates logging (context / pack rows), never a tensor the output depends on.
_BRANCH_READONLY = False


def set_branch_readonly(value=False):
    global _BRANCH_READONLY
    _BRANCH_READONLY = bool(value)


def set_motion_gamma(gamma=0.0, fluid=None, step_ratio=0.1):
    """Set DyMoS (inference only); ``step_ratio`` is paper lambda."""
    global _MOTION_GAMMA, _MOTION_FLUID, _MOTION_STEP_RATIO
    _MOTION_GAMMA = float(gamma)
    _MOTION_FLUID = fluid
    _MOTION_STEP_RATIO = float(step_ratio)


def set_denoise_step_fraction(value=None):
    """Expose the current within-block denoising fraction to attention.

    ``None`` denotes reference-cache initialization/final clean-cache refresh,
    where DyMoS must be disabled.  Zero is the first denoising evaluation.
    """
    global _DENOISE_STEP_FRACTION
    _DENOISE_STEP_FRACTION = None if value is None else float(value)


def _motion_gamma_for_step():
    if _DENOISE_STEP_FRACTION is None:
        return 0.0
    if _MOTION_STEP_RATIO <= 0.0:
        return 0.0
    return _MOTION_GAMMA if _DENOISE_STEP_FRACTION < _MOTION_STEP_RATIO else 0.0


def _dymos_attention(q, k, v, frame_seqlen, gamma):
    """SDPA with -gamma added to the logits of the first frame_seqlen (frame-0) keys."""
    b, Lq, n, d = q.shape
    Lk = k.shape[1]
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,n,L,d]
    bias = torch.zeros(Lk, device=q.device, dtype=qt.dtype)
    bias[:frame_seqlen] = -float(gamma)
    if _MOTION_FLUID is not None and _MOTION_FLUID.numel() == Lq:
        fw = _MOTION_FLUID.to(device=q.device, dtype=qt.dtype).view(1, 1, Lq, 1)
        attn_mask = bias.view(1, 1, 1, Lk) * fw                 # [1,1,Lq,Lk] region-gated
    else:
        attn_mask = bias.view(1, 1, 1, Lk)                      # broadcast over queries
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask)
    return out.transpose(1, 2)                                   # [b,Lq,n,d]


def _ref_bias_attention(q, k, v, off, frame_seqlen, gamma):
    """factorial.dymos_ref_scale: DyMoS (arXiv 2605.19398, Eq. 6) on a KV window. SDPA with the
    additive PRE-SOFTMAX bias  L~[i,j] = L[i,j] - gamma * phi(f(i)) * 1[j in source keys] *
    1[i not in source queries]  on the scaled logits L = QK^T / sqrt(d), uniform frame-wise schedule
    phi = 1 (the paper's default; Appendix D.2), for the source (reference) keys at window offset
    `off` (frame_seqlen tokens). Every query here belongs to a generated block, so the paper's
    frame-0-query exemption is the caller's `current_start_frame > 0` condition. The softmax is
    otherwise unchanged (no renormalisation beyond it). Unlike `_dymos_attention` (the Phase-1 port:
    bias on the FIRST frame_seqlen keys of the window), the bias sits at the source's actual window
    offset, which under Echo is not always 0."""
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,n,L,d]
    bias = torch.zeros(k.shape[1], device=q.device, dtype=qt.dtype)
    bias[off:off + frame_seqlen] = -float(gamma)
    out = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=bias.view(1, 1, 1, -1))
    return out.transpose(1, 2)                                   # [b,Lq,n,d]


def _frame_bias_attention(q, k, v, bias):
    """ fgs_soft: SDPA with a per-KEY additive pre-softmax bias `bias` [Lk] on the scaled logits
    QK^T / sqrt(d) (constant within each key frame: d * log(rho) for a history frame at age d latents, 0 for the
    source and the current block), broadcast over heads and queries. A constant bias c on all tokens of a frame
    multiplies that frame's softmax numerator mass by exp(c) and leaves the distribution WITHIN the frame unchanged."""
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,n,L,d]
    mask = bias.to(device=q.device, dtype=qt.dtype)
    mask = mask.view(1, 1, 1, -1) if mask.dim() == 1 else mask.view(1, 1, *mask.shape)   #: [Lq, Lk] locality
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.transpose(1, 2)                                   # [b,Lq,n,d]


def _sample_lol_thetas(num_heads, sigma, seed, base=10000.0):
    """Deterministic LoL Alg. 1 head bases: theta_h=theta0(1+sigma*U[-1,1])."""
    if sigma <= 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    epsilon = torch.empty(num_heads, dtype=torch.float64).uniform_(
        -1.0, 1.0, generator=generator)
    return base * (1.0 + float(sigma) * epsilon)


def _capture_anchor_attention(q, k, frame_seqlen, sink_tokens, current_start_frame):
    if not _ATTN_CAPTURE:
        return
    try:
        with torch.no_grad():
            d = q.shape[-1]
            qm = q.float().mean(dim=1, keepdim=True)                       # [b,1,n,d]
            s = torch.einsum('bqnd,bknd->bnqk', qm, k.float()) / (d ** 0.5)  # [b,n,1,Lk]
            a = s.softmax(dim=-1)[:, :, 0]                                  # [b,n,Lk]
            anchor = a[..., :frame_seqlen].sum(-1).mean().item()
            sink_n = max(frame_seqlen, int(sink_tokens))
            sink = a[..., :sink_n].sum(-1).mean().item()
            _ATTN_RECORDS.append({"frame": int(current_start_frame),
                                  "anchor_attn": float(anchor),
                                  "sink_attn": float(sink)})
    except Exception:
        pass


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 relativistic_rope=False,
                 ema_sink=False,
                 compression_alpha=0.99,
                 frame_seq_length=None,
                 rope_jitter_sigma=0.0,
                 rope_jitter_seed=0,
                 echo_memory=False,
                 echo_anchor_pool=18,
                 echo_anchor_frames=12,
                 echo_compressed_frames=3,
                 echo_recent_frames=6,
                 echo_anchor_stride=3,
                 echo_preserve_spatial=False,
                 echo_spectral_strength=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Steady-forcing flags (default off -> original cache path).
        # relativistic_rope: cache raw K and re-rope the whole window each step.
        # ema_sink: dual-sink + EMA compression of evicted tokens on rollover.
        # compression_alpha: EMA RETENTION a of the soft-sink state, applied once per evicting
        #   block: sink = a * sink + (1 - a) * evicted (see _cache_attention). The new-state
        #   mixing weight is 1 - a; factorial.ema_alpha_normalize rescales a per block size.
        self.relativistic_rope = relativistic_rope
        self.ema_sink = ema_sink
        self.compression_alpha = compression_alpha
        theta = _sample_lol_thetas(num_heads, rope_jitter_sigma, rope_jitter_seed)
        # Move with the module but never enter checkpoint state (training-free).
        self.register_buffer("rope_theta_by_head", theta, persistent=False)
        self.echo_memory = bool(echo_memory)
        self.echo_preserve_spatial = bool(echo_preserve_spatial)
        self.echo_spectral_strength = float(echo_spectral_strength)
        if not 0 <= self.echo_spectral_strength <= 1:
            raise ValueError('echo_spectral_strength must be in [0, 1]')
        if self.echo_spectral_strength and not (self.echo_memory and self.echo_preserve_spatial):
            raise ValueError('spectral arm requires spatial-corrected Echo')
        self.echo_anchor_pool = int(echo_anchor_pool)
        self.echo_anchor_frames = int(echo_anchor_frames)
        self.echo_compressed_frames = int(echo_compressed_frames)
        # Echo's R count includes the frame currently being evaluated.
        self.echo_recent_past = max(0, int(echo_recent_frames) - 1)
        self.echo_anchor_stride = int(echo_anchor_stride)
        self.reset_echo_memory()
        # B9: opt-in checkpoint-safe cache (CausalWanModel.set_checkpoint_safe_cache)
        self.checkpoint_safe_cache = False
        # Tokens per latent frame. 1560 is a WAN2.1 constant: at 480x832 the VAE
        # (stride 4,8,8) gives 60x104 latents, which patch (1,2,2) turns into 30*52=1560
        # tokens. Wan2.2-TI2V-5B has stride (4,16,16) -> 30x52 latents -> 15*26=390
        # tokens, so the same expression would size the attention window 4x too large and
        # the KV-cache/eviction arithmetic would silently address the wrong frames.
        # Resolution order: explicit argument > env override > the Wan2.1 default.
        if frame_seq_length is None:
            frame_seq_length = int(os.environ.get("WAN_FRAME_SEQ_LENGTH", 1560))
        self.frame_seq_length = frame_seq_length
        self.max_attention_size = (
            21 * frame_seq_length if local_attn_size == -1
            else local_attn_size * frame_seq_length)

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def reset_echo_memory(self):
        self._echo_early_k, self._echo_early_v, self._echo_early_ids = [], [], []
        self._echo_recent_k, self._echo_recent_v, self._echo_recent_ids = [], [], []
        self._echo_compressed_k = self._echo_compressed_v = None
        self._echo_compressed_spatial = None
        self._echo_compressed_origin = None
        self._echo_compressed_adj = None      # H2: persistent admission adjustment of compressed tokens
        self._echo_q_center = None
        self._echo_q_count = 0
        self._echo_spectral_calls = 0
        self._echo_spectral_first_frame = None
        # factorial.echo_cadence_latents (nfpb 1 only): clean K/V of the latents of the
        # current, not yet committed cadence group. Empty (no-op) unless the cadence is set.
        self._echo_pending_q, self._echo_pending_k, self._echo_pending_v = [], [], []
        self._echo_pending_ids = []

    def _echo_anchor_indices(self, frame_index):
        """Bidirectional rolling selection from the early-frame pool."""
        pool = len(self._echo_early_k)
        take = min(self.echo_anchor_frames, pool)
        if take == 0:
            return []
        update = max(0, frame_index - self.echo_anchor_pool) // max(1, self.echo_anchor_stride)
        start = (update * self.echo_anchor_stride) % pool
        order = [(start + i) % pool for i in range(take)]
        if update % 2:
            order.reverse()
        return order

    def _echo_pin_anchor_indices(self, frame_index, frame_seqlen):
        """factorial.echo_pin_source / echo_pin_latent; opt-in, never called on the
        default path). The pinned latent p takes ONE of the `echo_anchor_frames` anchor slots for the
        whole rollout; the other A - 1 slots rotate with the unchanged formula of
        `_echo_anchor_indices` (same update / stride / reversal) over the early pool MINUS the entry
        that IS the pinned frame (a 1-frame entry: the I2V source entry for p = 0; latent p's own
        entry at nfpb 1). A pinned latent inside a multi-frame entry (p >= 1 at nfpb 3) leaves that
        entry in the rotation (the pinned frame may then be attended twice)."""
        p = int(self.echo_pin_latent)
        gen = [i for i, (tick, ek) in enumerate(zip(self._echo_early_ids, self._echo_early_k))
               if not (tick == p and ek.shape[1] == frame_seqlen)]
        pool = len(gen)
        take = min(self.echo_anchor_frames - 1, pool)
        if take <= 0:
            return []
        update = max(0, frame_index - self.echo_anchor_pool) // max(1, self.echo_anchor_stride)
        start = (update * self.echo_anchor_stride) % pool
        order = [gen[(start + i) % pool] for i in range(take)]
        if update % 2:
            order.reverse()
        return order

    def _echo_pin_kv(self, frame_seqlen):
        """K/V of the pinned latent (ONE latent frame) sliced out of whichever early Echo entry holds
        it: tick 0 = the source latent (I2V: a 1-frame entry); a generated latent p is the
        (p - tick)-th frame of its entry. None until that latent has been committed. Returns
        (k, v, tick, whole_entry)."""
        p = int(self.echo_pin_latent)
        for tick, ek, ev in zip(self._echo_early_ids, self._echo_early_k, self._echo_early_v):
            n = ek.shape[1] // frame_seqlen
            if tick <= p < tick + n:
                o = (p - tick) * frame_seqlen
                return ek[:, o:o + frame_seqlen], ev[:, o:o + frame_seqlen], tick, n == 1
        return None

    # --- C2 / C3 / DyMoS baseline (opt-in; none of these attributes exists on the
    # default path, so every branch below is skipped and the default is bitwise unchanged) -----
    #   source_entries  {"sharp": (k, v), "soft": (k, v)}: TWO precomputed K/V entries of the I2V
    #                   source latent in this policy's STORED form (sink: key roped at frame 0 as
    #                   held in cache slot 0, raw value; Echo: raw key / value like early entry 0).
    #                   "sharp" is the entry the default path stores; "soft" is the same entry
    #                   computed from a low-passed source latent, or the sharp one with the values
    #                   down-weighted. Set by CausalInferencePipeline._prepare_source_entries.
    #   source_read     "soft" | "sharp": which entry THIS evaluation reads (set per denoising
    #                   evaluation by the pipeline from the current timestep; "sharp" at the
    #                   clean refresh). Mechanism audit: what is filtered is the
    #                   reference K/V ENTRY as read by attention -- not the input latent, not the
    #                   attention output, not the generated latents; it is indexed by noise level
    #                   (timestep), and it is never persisted: the cache / Echo memory only ever
    #                   hold the sharp entry (the soft one is substituted into a fresh window
    #                   tensor per call and discarded).
    #   source_role     bool [frame_seqlen] (True = rigid support, False = moving material) or
    #                   None: C3 per-query routing. Support queries read the SHARP entry at every
    #                   step; material queries read the scheduled (`source_read`) entry. Online,
    #                   model-intrinsic (pipeline._role_routing_update); never an evaluation mask.
    #   dymos_ref_scale / dymos_active   gamma of `_ref_bias_attention` and whether this
    #                   evaluation is one of the first `early_steps` of its block.
    def _dymos_ref_gamma(self, current_start_frame):
        s = getattr(self, 'dymos_ref_scale', None)
        if s is None or current_start_frame == 0 or not getattr(self, 'dymos_active', False):
            return 0.0
        return float(s)

    def _source_routed_attention(self, rq, rk, vv, off, frame_seqlen, gamma, soft):
        """Attention over the window [rk | vv] whose SHARP source entry occupies tokens
        [off, off + frame_seqlen), with the C2 entry selection and the C3 per-query routing.
        `soft` = (k, v) of the soft entry in the same (roped) form as rk's source part, or None.
        Exact two-pass implementation: attention is per-query, so the support queries and the
        material queries are gathered along the token axis, each set attends its own window
        (sharp source / scheduled source; everything else identical), and the outputs are
        scattered back. Cost: the same number of query rows as one pass (two kernel launches),
        one window copy with the soft part when it is read, and a gather / scatter of the queries;
        no extra keys, no attention mask. `gamma` != 0 adds the DyMoS source-key bias in every
        pass (`_ref_bias_attention`); otherwise the unchanged `attention` kernel is used."""
        read = getattr(self, 'source_read', 'sharp')
        role = getattr(self, 'source_role', None)
        use_soft = soft is not None and read == 'soft'

        def run(q, k, v):
            if gamma:
                return _ref_bias_attention(q, k, v, off, frame_seqlen, gamma)
            return attention(q, k, v)

        if not use_soft:
            return run(rq, rk, vv)
        k_soft = torch.cat([rk[:, :off], soft[0].type_as(rk), rk[:, off + frame_seqlen:]], dim=1)
        v_soft = torch.cat([vv[:, :off], soft[1].type_as(vv), vv[:, off + frame_seqlen:]], dim=1)
        if role is None:
            return run(rq, k_soft, v_soft)
        n_q = rq.shape[1] // frame_seqlen
        support = role.to(rq.device).repeat(n_q)                  # tiled over the block's frames
        idx_s = support.nonzero(as_tuple=True)[0]
        idx_m = (~support).nonzero(as_tuple=True)[0]
        if idx_m.numel() == 0:
            return run(rq, rk, vv)
        if idx_s.numel() == 0:
            return run(rq, k_soft, v_soft)
        out_m = run(rq.index_select(1, idx_m), k_soft, v_soft)
        out_s = run(rq.index_select(1, idx_s), rk, vv)
        out = torch.empty(rq.shape, dtype=out_m.dtype, device=rq.device)
        out.index_copy_(1, idx_m, out_m)
        out.index_copy_(1, idx_s, out_s)
        return out

    # --- C1 budget accounting (opt-in `factorial.context_accounting`; layer 0 only;
    # logging: nothing the output depends on is touched) -------------------------------------
    def _source_mass(self, rq, rk, off, frame_seqlen, idx, gamma):
        """Mean softmax mass on the source keys [off, off + frame_seqlen) over the query rows `idx`
        (a detached float32 sub-batch of at most 256 rows: scores against the full window, the DyMoS
        bias included when active). Never on the kernel path."""
        if idx.numel() == 0:
            return None
        with torch.no_grad():
            q = rq.index_select(1, idx)[0].float()                        # [m, n, d]
            s = torch.einsum("qnd,knd->nqk", q, rk[0].float()) / math.sqrt(q.shape[-1])
            if gamma:
                s[:, :, off:off + frame_seqlen] -= float(gamma)
            pb = getattr(self, '_pack_bias', None)
            if pb is not None and pb.numel() == s.shape[-1]:
                s = s + pb.to(device=s.device, dtype=s.dtype)            # fgs_soft: the read as attended
            p = s.softmax(-1)
            return float(p[:, :, off:off + frame_seqlen].sum(-1).mean())

    # --- training-free PORTS of two cache-packing mechanisms (opt-in; read-time only;
    # nothing written into history; the default path never reaches these: attributes absent) -------
    #   packcache_decay {rho, rule, window}: PackCache (arXiv 2601.04359) cross-frame decay budget.
    #       Per attended HISTORY frame at temporal distance d = cur - latent_id (in latents, the paper's
    #       AR unit) keep t_d = round(frame_seqlen * b_d) tokens with b_d = rho^min(d, W-1); for rho = 1/2
    #       this is the paper's closed form b_d = 2^-min(d, W-1) ([1], [1/2, 1/2], [1/2, 1/4, 1/4] ...;
    #       t_d = B_one * b_d with B_one = one frame's tokens). The source frame keeps every token at
    #       every block (condition anchoring: "contributions do not decay with temporal distance";
    #       on Echo that is pinsrc, on the sinks slot 0); the current block's own frames are full.
    #       Frames whose budget rounds to 0 are dropped; compressed / blended frames (no latent id) get
    #       the minimum budget b_{W-1}. INTRA-FRAME RULE: the paper does not state which tokens are
    #       kept ("unmasked and attention-relevant tokens"; so this is OUR pre-registered
    #       choice, not the paper's: `rule: keynorm` (default) keeps the t_d tokens with the largest
    #       L2 key norm (RoPE is a rotation, so the norm of the roped window key equals the raw key
    #       norm); `rule: stride` keeps a fixed spatial stride (token (i * frame_seqlen) // t_d for
    #       i < t_d), content-independent. Kept tokens keep their (h, w) and temporal RoPE (the window
    #       is subsampled AFTER RoPE: the paper's "spatially preserving position embedding", Eq. 9).
    #       What does not transfer: the discrete-token masked decoding, the in-context text
    #       anchor (we have cross-attention), the speed claim.
    #   fp_pool {full, half, quarter, beyond}: FramePack (arXiv 2504.12626) geometric context packing as
    #       a K/V-pooling ANALOGUE. By temporal distance d: the most recent `full` (1) history frame at
    #       1x, the next `half` (2) frames average-pooled over 2x2 token neighbourhoods (1/4 tokens),
    #       the next `quarter` (16) over 4x4 (1/16 tokens; the F1 demo's 1 / 2 / 16 at 1x / 2x / 4x),
    #       anything older `beyond: drop` (FramePack's fixed context length) or `quarter` (4x4 too);
    #       source and current block at 1x. Pooled over the ROPED keys and the values at read time:
    #       the mean of rotated keys is NOT the rotation of the mean key (approximation, stated);
    #       pooled tokens carry no new position. ceil_mode pooling (30 x 52 -> 15 x 26 at 2x, 8 x 13 at
    #       4x; partial windows average their valid tokens). What does not transfer: FramePack
    #       compresses INPUTS through learned per-rate patchify kernels (we pool frozen K/V), and its
    #       anti-drifting (inverted / endpoint-planned bidirectional sampling, F1's trained regulation)
    #       cannot exist in a causal few-step student. This is "FramePack-schedule K/V pooling", never
    #       "FramePack".
    @staticmethod
    def _loc_grid(h, w, device):
        """: (row, col) of every token of an h x w frame, row-major, float32 [h*w, 2]."""
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        return torch.stack([yy.flatten(), xx.flatten()], dim=1).float()

    @staticmethod
    def _loc_pool_pos(h, w, r, device):
        """: centre (row, col) of every r x r ceil-mode pooling cell, in `_pool_tokens` order."""
        cy = torch.tensor([(i * r + min(i * r + r, h) - 1) / 2.0 for i in range(-(-h // r))], device=device)
        cx = torch.tensor([(j * r + min(j * r + r, w) - 1) / 2.0 for j in range(-(-w // r))], device=device)
        yy, xx = torch.meshgrid(cy, cx, indexing="ij")
        return torch.stack([yy.flatten(), xx.flatten()], dim=1).float()

    def _locality_mask(self, n_q, frame_seqlen, grid_sizes, device):
        """ fixed-camera locality prior: additive pre-softmax bias [n_q, Lk] = -((max(0, dist - radius)) /
        sigma)^2 between each current-block query (grid position) and each source / history key (`_pack_pos`; current
        block keys and keys without a position: 0), plus the per-key `_pack_bias` when present."""
        loc = self.locality
        h, w = int(grid_sizes[0][1]), int(grid_sizes[0][2])
        qpos = self._loc_grid(h, w, device).repeat(n_q // frame_seqlen, 1)
        kpos = self._pack_pos.to(device)
        valid = ~torch.isnan(kpos[:, 0])
        d = torch.cdist(qpos, torch.nan_to_num(kpos, nan=0.0))
        bias = -(torch.relu(d - float(loc["radius"])) / float(loc["sigma"])) ** 2
        bias = bias * valid.to(bias.dtype)[None, :]
        pb = getattr(self, '_pack_bias', None)
        if pb is not None:
            bias = bias + pb.to(device=device, dtype=bias.dtype)[None, :]
        return bias

    def _pool_tokens(self, x, h, w, r):
        """Average-pool a frame's tokens [b, h*w, n, d] over r x r token neighbourhoods (ceil_mode:
        the last window of a non-divisible grid averages its valid tokens) -> [b, h' * w', n, d]."""
        b, L, n, d = x.shape
        y = x.reshape(b, h, w, n * d).permute(0, 3, 1, 2)
        if y.dtype not in (torch.float32, torch.float64):
            y = y.float()                      # pool half precisions in float32
        y = torch.nn.functional.avg_pool2d(y, kernel_size=r, stride=r, ceil_mode=True)
        return y.permute(0, 2, 3, 1).reshape(b, -1, n, d).to(x.dtype)

    def _sink_window_ids(self, cur, nq, n_win, n_buf=None):
        """Latent id of every frame slot of the sink window (None = an EMA blend): before the first
        eviction latents 0..cur+k-1; afterwards [sink slots][the most recent latents][current].
        `n_buf` (frames in the KV buffer) > n_win when a per-layer read window (factorial.
        local_layer_sizes, reads only the LAST n_win buffer frames."""
        n_buf = n_win if n_buf is None else int(n_buf)
        s = int(self.sink_size)
        if cur + nq == n_buf:
            ids = list(range(n_buf))
        else:
            sink_ids = ([0] + [None] * (s - 1)) if (self.ema_sink and s > 1) else list(range(s))
            rolled = n_buf - s - nq
            ids = sink_ids + [cur - rolled + j for j in range(rolled)] + [cur + j for j in range(nq)]
        return ids[len(ids) - n_win:]

    def _pack_window(self, rk, vv, seq, cur, frame_seqlen, grid_sizes, keep_full=()):
        """Apply packcache_decay / fp_pool to the attended window (roped keys rk, values vv, one
        frame slot per entry of `seq`). Returns (k, v, source token offset or None,
        [(latent id, kept tokens)] in window order). Read-time only."""
        dec = getattr(self, 'packcache_decay', None)
        pool = getattr(self, 'fp_pool', None)
        fg = getattr(self, 'forget', None)
        src_anchor = not getattr(self, 'pack_source_decay', False)   # fg_nosrc: source decayed too
        W = self.max_attention_size // frame_seqlen
        h, w = int(grid_sizes[0][1]), int(grid_sizes[0][2])
        ks, vs, kept = [], [], []
        off, src_off = 0, None
        frames_rec = [] if fg is not None else None
        src_v = None
        tether_on = fg is not None and fg.get("prev_tether") is not None
        tether_conf = getattr(self, "forget_static_conf", None) if tether_on else None
        tether_source = None
        if tether_on and tether_conf is not None:
            if seq.count(0) != 1:
                raise RuntimeError("static value tether requires exactly one source frame")
            sj = seq.index(0)
            tether_source = vv[:, sj * frame_seqlen:(sj + 1) * frame_seqlen]
        # (opt-in forget keys; absent on every earlier arm, so these branches are never entered there):
        #   soft    keep every token, per-frame additive logit bias d * log(rho) (fgs_soft)
        #   sticky  a history latent's kept set is ranked ONCE (first entry into the decayed region) and only shrinks
        fg_soft = fg is not None and bool(fg.get("soft", False))
        fg_sticky = fg is not None and bool(fg.get("sticky", False))
        # (opt-in; absent on every earlier arm): how the NEWEST history latent (id cur - 1) is read
        #   packcache prev_mode  pool2: 2x2 average-pooled (390 of 1560 tokens, every grid cell covered)
        #                        soft:  every token kept, frame logit bias log(round(1560 rho) / 1560) (mass = the budget)
        #   forget prev_floor    the newest latent keeps max(B, round(floor * 1560)) tokens by fixed spatial stride
        dec_prev_mode = None if dec is None else dec.get("prev_mode")
        fg_prev_floor = None if fg is None else fg.get("prev_floor")
        fg_prev_mode = None if fg is None else fg.get("prev_mode")
        soft_bias = [] if (fg_soft or dec_prev_mode == "soft") else None   # (tokens, bias) per appended frame
        fb = 0.0
        # locality prior (opt-in): grid position (row, col; token units) of every kept key, None = no bias
        loc_on = getattr(self, 'locality', None) is not None
        poss = [] if loc_on else None
        full_pos = self._loc_grid(h, w, rk.device) if loc_on else None
        if fg_sticky and getattr(self, '_fg_sticky', None) is None:
            self._fg_sticky = {}                               # latent id -> ranked kept token indices (per video)
        if fg is not None:
            newest = cur + sum(1 for f in seq if f is not None and f >= cur) - 1
            ref_mode = fg.get("reference", "source") if fg.get("redundancy") is not None else None
            if fg.get("redundancy") is not None and 0 in seq:
                #: the source frame's RAW values (never roped; the cache stores them raw in every slot)
                j0 = seq.index(0)
                src_v = vv[:, j0 * frame_seqlen:(j0 + 1) * frame_seqlen]
            ref_fixed, ref_name = src_v, "source"
            if ref_mode == "pshuf":
                #: this scene's source V at permuted token positions (right statistics, wrong places)
                perm = getattr(self, 'forget_ref_perm', None)
                ref_fixed = None if (src_v is None or perm is None) else src_v.index_select(1, perm.to(src_v.device))
                ref_name = "source_pshuf"
            elif ref_mode == "xref":
                #: another dev scene's source V, captured once per video (never in the real cache)
                rv = getattr(self, 'forget_ref_v', None)
                ref_fixed = None if rv is None else rv.to(device=vv.device, dtype=vv.dtype)
                ref_name = "xref"
            elif ref_mode == "adjacent":
                #: the next-newer frame with a latent id in the window (current block / recent included)
                win_ids = sorted({f for f in seq if f is not None})
                win_pos = {f: i for i, f in enumerate(seq) if f is not None}
                ref_fixed, ref_name = None, "adjacent"
        for j, fid in enumerate(seq):
            kj = rk[:, j * frame_seqlen:(j + 1) * frame_seqlen]
            vj = vv[:, j * frame_seqlen:(j + 1) * frame_seqlen]
            fb = 0.0
            kp = None
            if (fid == 0 and src_anchor) or (fid is not None and fid >= cur) or j in keep_full:
                # condition anchor / current block / (ema_sink, pack_soft_sink: full) the EMA soft-sink slots
                keep_k, keep_v = kj, vj
            elif fg is not None:
                # material-aware forgetting: per-frame budget = the uniform rho budget (or full inside
                # the recent window, in latents from the NEWEST latent), split between the two roles in proportion to
                # N_role * rho_role0^x (x = age in latents = video seconds / 0.25 s, capped like the uniform budget)
                d = min(cur - fid if fid is not None else W - 1, W - 1)
                recent = fid is not None and fid > newest - int(fg["recent_latents"])
                if fg_soft:
                    # fgs_soft: every token kept; logit bias d * log(rho) on the whole frame (0 inside the
                    # recent window), i.e. the frame's softmax mass is scaled by rho^d (the hard budget's keep FRACTION),
                    # its within-frame distribution untouched; no token is dropped, nothing selected
                    lb = 0.0 if recent else d * math.log(float(fg["rho"]))
                    frames_rec.append(dict(fid=fid, age_latents=d, recent=bool(recent), budget=frame_seqlen,
                                           kept=frame_seqlen, selection="soft", bias=lb, mass_scale=math.exp(lb),
                                           budget_equiv=frame_seqlen if recent
                                           else int(frame_seqlen * (float(fg["rho"]) ** d) + 0.5)))
                    ks.append(kj)
                    vs.append(vj)
                    kept.append((fid, frame_seqlen))
                    soft_bias.append((frame_seqlen, lb))
                    if loc_on:
                        poss.append(full_pos)
                    off += frame_seqlen
                    continue
                B = frame_seqlen if recent else int(frame_seqlen * (float(fg["rho"]) ** d) + 0.5)
                # H1: role_older false -> older latents ignore the role map (roles-none selection); the map then
                # only drives the newest-latent read (prev_split)
                role = getattr(self, 'forget_role', None) if fg.get("role_older", True) else None
                ent = dict(fid=fid, age_latents=d, recent=bool(recent), budget=int(B))
                if B <= 0:
                    if fg_sticky:
                        self._fg_sticky.pop(fid, None)          # budgets only shrink with age: never needed again
                    ent.update(kept=0, kept_material=0, kept_rigid=0)
                    frames_rec.append(ent)
                    continue
                if B >= frame_seqlen:
                    keep_k, keep_v = kj, vj
                    nm = None if role is None else int(role.sum())
                    ent.update(kept=frame_seqlen, kept_material=nm, kept_rigid=None if nm is None else frame_seqlen - nm)
                else:
                    ref_v, ref_id = (ref_fixed, ref_name) if fg.get("redundancy") is not None else (None, None)
                    if ref_mode == "adjacent" and fid is not None:
                        nxt = next((f for f in win_ids if f > fid), None)
                        if nxt is not None:
                            jn = win_pos[nxt]
                            ref_v, ref_id = vv[:, jn * frame_seqlen:(jn + 1) * frame_seqlen], f"adj:{nxt}"
                    pool_keep = None
                    if fg.get("prev_split", False) and fid is not None and fid == cur - 1:
                        # H1 motion-aware read of the newest history latent: every token the online volatility
                        # map marks moving OR uncertain (log10 var > threshold - band) is kept exact; the remaining
                        # (static) tokens keep prev_static_frac of their count at fixed stride. Before the first map,
                        # the newest latent is read whole (uncertainty defaults to dense recent context).
                        dense = getattr(self, 'forget_prev_dense', None)
                        if dense is None:
                            idx = torch.arange(frame_seqlen, device=kj.device)
                            ent.update(kept=frame_seqlen, kept_material=None, kept_rigid=None, selection="prev_split_nomap")
                        else:
                            dense = dense.to(kj.device)
                            # controls (opt-in; default "map" is bitwise the locked read): inverted = dense where the
                            # map is static and strided where it moves; random = the same per-latent token COUNT as the map
                            # read, at uniformly random positions redrawn every block (seeded per video and block)
                            sel = fg.get("prev_select", "map")
                            if sel == "inverted":
                                dense = ~dense
                            dyn = dense.nonzero(as_tuple=True)[0]
                            st = (~dense).nonzero(as_tuple=True)[0]
                            ns = int(st.numel() * float(fg.get("prev_static_frac", 0.3)) + 0.5)
                            if sel == "random":
                                n_keep = int(dyn.numel()) + ns
                                g = torch.Generator(device="cpu").manual_seed(
                                    int(getattr(self, 'forget_rand_seed', 0)) * 1_000_003 + int(cur) * 7919 + 101)
                                idx = torch.randperm(frame_seqlen, generator=g)[:n_keep].to(kj.device).sort().values
                                ent.update(kept=int(idx.numel()), kept_material=None, kept_rigid=None, selection="prev_random")
                            else:
                                idx = torch.cat([dyn, self._forget_select(kj, st, ns, "stride")]).sort().values
                                if tether_on and tether_conf is not None:
                                    # Full coverage on confident static tokens; optional full band coverage.
                                    interior = tether_conf.to(kj.device) > 0
                                    extra = (~dense) if fg.get("prev_tether_band") is not None else interior
                                    idx = torch.unique(torch.cat([idx, extra.nonzero(as_tuple=True)[0]]), sorted=True)
                                ent.update(kept=int(idx.numel()), kept_material=int(dyn.numel()), kept_rigid=ns,
                                           selection="prev_split" if sel == "map" else "prev_inverted")
                    elif fg_prev_mode is not None and fid is not None and fid == cur - 1:
                        # round 5: the newest history latent as its 2x2-pooled summary (pool2), or that summary
                        # plus its B exact tokens by fixed stride (pool2_stride); older latents keep this path's selection
                        pool_keep = [self._pool_tokens(kj, h, w, 2)], [self._pool_tokens(vj, h, w, 2)]
                        if fg_prev_mode == "pool2_stride":
                            idx = (torch.arange(B, device=kj.device) * frame_seqlen) // B
                            pool_keep[0].append(kj.index_select(1, idx))
                            pool_keep[1].append(vj.index_select(1, idx))
                        nk = sum(int(x.shape[1]) for x in pool_keep[0])
                        ent.update(kept=nk, kept_material=None, kept_rigid=None, selection=f"prev_{fg_prev_mode}")
                    elif fg_prev_floor is not None and fid is not None and fid == cur - 1:
                        #: the newest history latent - spatially uniform coverage (fixed stride) at >= the floor;
                        # older latents keep this path's selection (key norm / source-consistent)
                        B = max(B, int(frame_seqlen * float(fg_prev_floor) + 0.5))
                        idx = (torch.arange(B, device=kj.device) * frame_seqlen) // B
                        ent.update(kept=B, budget=int(B), kept_material=None, kept_rigid=None, selection="prev_stride")
                    elif fg.get("stratify", False):
                        # stratified selection: the frame is cut into B raster spans of 3-4 (newest latent) or
                        # more tokens; ONE token per span, the best of this path's ranking (most source-consistent for
                        # srcred, ties by key norm; largest key norm for roles none) - full spatial coverage
                        if fg.get("redundancy") is not None and ref_v is not None:
                            order, _ = self._srcred_order(kj, vj, ref_v, fg["redundancy"])
                        else:
                            order = torch.sort(-kj[0].float().flatten(1).norm(dim=-1), stable=True).indices
                        idx = self._stratified_pick(order, frame_seqlen, B)
                        ent.update(kept=B, kept_material=None, kept_rigid=None,
                                   selection="strat_srcred" if fg.get("redundancy") is not None else "strat_keynorm")
                    elif fg_sticky and fid is not None:
                        # fgs_sticky / fgs_sticky_srcinv: rank once, at the first block in which this latent
                        # is attended with a partial budget (key norm, or source similarity with ties by key norm), keep
                        # that top-B ranked list; afterwards the kept set is its first B' <= B entries (nested over time)
                        st = self._fg_sticky.get(fid)
                        srcrank = fg.get("redundancy") is not None and ref_v is not None
                        if st is None:
                            if srcrank:
                                order, _ = self._srcred_order(kj, vj, ref_v, fg["redundancy"])
                            else:
                                order = torch.sort(-kj[0].float().flatten(1).norm(dim=-1), stable=True).indices
                            st = (int(cur), order[:B].clone())
                            self._fg_sticky[fid] = st
                        if B > st[1].numel():
                            raise RuntimeError(f"sticky forgetting: budget of latent {fid} grew ({st[1].numel()} -> {B})")
                        idx = st[1][:B].sort().values
                        ent.update(kept=B, kept_material=None, kept_rigid=None, sticky_since=st[0],
                                   selection=f"sticky_srcred_{fg['redundancy']}" if srcrank else "sticky_keynorm",
                                   **({"reference": ref_id} if srcrank else {}))
                        if srcrank and getattr(self, 'context_log', None) is not None and _DENOISE_STEP_FRACTION == 0.0 \
                                and not _BRANCH_READONLY:
                            red = self._source_redundancy(vj, ref_v)
                            drop = torch.ones(frame_seqlen, dtype=torch.bool, device=red.device)
                            drop[idx] = False
                            nd = frame_seqlen - B
                            ent.update(red_kept_mean=float(red[idx].mean()),
                                       red_dropped_mean=float(red[drop].mean()) if nd > 0 else None,
                                       red_kept_sum=float(red[idx].sum()), red_dropped_sum=float(red[drop].sum()),
                                       n_dropped=nd)
                    elif fg.get("redundancy") is not None and ref_v is not None:
                        idx, red = self._srcred_select(kj, vj, ref_v, B, fg["redundancy"])
                        ent.update(kept=B, kept_material=None, kept_rigid=None, selection=f"srcred_{fg['redundancy']}",
                                   reference=ref_id)
                        if getattr(self, 'context_log', None) is not None and _DENOISE_STEP_FRACTION == 0.0 \
                                and not _BRANCH_READONLY:
                            # logging only (layer 0, first evaluation of the block): host syncs kept off the hot path
                            drop = torch.ones(frame_seqlen, dtype=torch.bool, device=red.device)
                            drop[idx] = False
                            nd = frame_seqlen - B
                            ent.update(red_kept_mean=float(red[idx].mean()),
                                       red_dropped_mean=float(red[drop].mean()) if nd > 0 else None,
                                       red_kept_sum=float(red[idx].sum()), red_dropped_sum=float(red[drop].sum()),
                                       n_dropped=nd)
                    elif role is None:
                        idx = self._forget_select(kj, torch.arange(frame_seqlen, device=kj.device), B, fg["rule"])
                        ent.update(kept=B, kept_material=None, kept_rigid=None,
                                   **({"selection": "keynorm_fallback_no_source"} if fg.get("redundancy") else {}))
                    else:
                        ri = getattr(self, '_forget_role_idx', None)     # role token indices, cached per map version
                        if ri is None or ri[0] != getattr(self, 'forget_map_version', 0) or ri[1].device != kj.device:
                            mat = role.to(kj.device)
                            ri = (getattr(self, 'forget_map_version', 0), mat.nonzero(as_tuple=True)[0],
                                  (~mat).nonzero(as_tuple=True)[0])
                            self._forget_role_idx = ri
                        cm, cr = ri[1], ri[2]
                        km, kr = self._forget_split(B, int(cm.numel()), int(cr.numel()), d, fg)
                        idx = torch.cat([self._forget_select(kj, cm, km, fg["rule"]),
                                         self._forget_select(kj, cr, kr, fg["rule"])]).sort().values
                        ent.update(kept=km + kr, kept_material=km, kept_rigid=kr, n_material=int(cm.numel()),
                                   rho_material_eff=(km / cm.numel()) ** (1.0 / d) if d > 0 and cm.numel() else None,
                                   rho_rigid_eff=(kr / cr.numel()) ** (1.0 / d) if d > 0 and cr.numel() else None)
                    if pool_keep is not None:
                        keep_k, keep_v = torch.cat(pool_keep[0], dim=1), torch.cat(pool_keep[1], dim=1)
                        if loc_on:
                            kp = torch.cat([self._loc_pool_pos(h, w, 2, rk.device)]
                                           + ([full_pos.index_select(0, idx)] if fg_prev_mode == "pool2_stride" else []))
                        idx = None
                    else:
                        keep_k, keep_v = kj.index_select(1, idx), vj.index_select(1, idx)
                        if tether_on and tether_conf is not None and fid == cur - 1:
                            from .static_fidelity import tether_values
                            keep_v = tether_values(vj, tether_source, idx, self.forget_prev_dense,
                                                   tether_conf, fg["prev_tether"], fg.get("prev_tether_band"))
                            ent.update(selection="static_value_tether", kept_rigid=int(idx.numel()) - int(dyn.numel()))
                        if loc_on:
                            kp = full_pos.index_select(0, idx)
                    tr = getattr(self, '_pack_trace', None)
                    if tr is not None:                           # tests only (absent in production): kept indices
                        tr.append((int(cur), fid, None if idx is None else idx.tolist()))
                frames_rec.append(ent)
            elif dec is not None:
                d = min(cur - fid if fid is not None else W - 1, W - 1)
                t = int(frame_seqlen * (float(dec["rho"]) ** d) + 0.5)
                # prev_floor (opt-in; absent on every earlier arm): the NEWEST history latent (id cur - 1)
                # keeps at least round(floor * frame_seqlen) tokens, chosen by the fixed spatial stride (uniform coverage)
                prev = dec.get("prev_floor") is not None and fid is not None and fid == cur - 1
                if prev:
                    t = max(t, int(frame_seqlen * float(dec["prev_floor"]) + 0.5))
                pmode = dec_prev_mode if (fid is not None and fid == cur - 1) else None
                if pmode == "soft" and 0 < t < frame_seqlen:
                    # fgs_prevsoft: the whole newest latent (no spatial holes) with the hard budget's MASS share
                    keep_k, keep_v, fb = kj, vj, math.log(t / frame_seqlen)
                elif pmode == "pool2":
                    # fgs_prevpool: the newest latent 2x2-pooled - every grid cell covered at a quarter resolution
                    keep_k, keep_v = self._pool_tokens(kj, h, w, 2), self._pool_tokens(vj, h, w, 2)
                    if loc_on:
                        kp = self._loc_pool_pos(h, w, 2, rk.device)
                elif pmode == "pool2_stride" and 0 < t < frame_seqlen:
                    # fgs_poolstride: the newest latent as its 2x2-pooled summary PLUS its t exact tokens by stride
                    idx = (torch.arange(t, device=kj.device) * frame_seqlen) // t
                    keep_k = torch.cat([self._pool_tokens(kj, h, w, 2), kj.index_select(1, idx)], dim=1)
                    keep_v = torch.cat([self._pool_tokens(vj, h, w, 2), vj.index_select(1, idx)], dim=1)
                    if loc_on:
                        kp = torch.cat([self._loc_pool_pos(h, w, 2, rk.device), full_pos.index_select(0, idx)])
                elif dec["rule"] == "pool":
                    # fgs_pooldecay: multi-resolution decay - a history latent at age d is 2^d x 2^d average-pooled
                    # (d capped at 5): 390, 104, 28, 8, 2 ... tokens instead of dropping tokens; nothing dropped
                    r = 2 ** min(d, 5)
                    keep_k, keep_v = self._pool_tokens(kj, h, w, r), self._pool_tokens(vj, h, w, r)
                    if loc_on:
                        kp = self._loc_pool_pos(h, w, r, rk.device)
                elif t <= 0:
                    continue                                      # budget rounds to no token: dropped
                elif t >= frame_seqlen:
                    keep_k, keep_v = kj, vj
                else:
                    if dec["rule"] == "stride" or prev:
                        idx = (torch.arange(t, device=kj.device) * frame_seqlen) // t
                    else:                                         # keynorm
                        norms = kj[0].float().flatten(1).norm(dim=-1)
                        idx = norms.topk(t).indices.sort().values
                    keep_k, keep_v = kj.index_select(1, idx), vj.index_select(1, idx)
                    if loc_on:
                        kp = full_pos.index_select(0, idx)
            else:                                                 # fp_pool
                d = cur - fid if fid is not None else None
                f1, f2, f4 = int(pool["full"]), int(pool["half"]), int(pool["quarter"])
                if d is not None and d <= f1:
                    r = 1
                elif d is not None and d <= f1 + f2:
                    r = 2
                elif d is not None and d <= f1 + f2 + f4:
                    r = 4
                elif pool["beyond"] == "quarter":
                    r = 4
                else:
                    continue                                      # beyond the fixed context: dropped
                if r == 1:
                    keep_k, keep_v = kj, vj
                else:
                    keep_k, keep_v = self._pool_tokens(kj, h, w, r), self._pool_tokens(vj, h, w, r)
                    if loc_on:
                        kp = self._loc_pool_pos(h, w, r, rk.device)
            if fid == 0 and int(keep_k.shape[1]) == frame_seqlen:      # source offset only while it is whole
                src_off = off
            ks.append(keep_k)
            vs.append(keep_v)
            kept.append((fid, int(keep_k.shape[1])))
            if loc_on:
                n_k = int(keep_k.shape[1])
                if fid is not None and fid >= cur:
                    kp = torch.full((n_k, 2), float("nan"), device=rk.device)   # current block: unbiased (global)
                elif kp is None:
                    if n_k != frame_seqlen:
                        raise RuntimeError(f"locality: no key positions for a thinned frame ({fid}, {n_k} tokens)")
                    kp = full_pos
                poss.append(kp)
            off += int(keep_k.shape[1])
            if soft_bias is not None:
                soft_bias.append((int(keep_k.shape[1]), fb))    # 0 (unbiased) except the fgs_prevsoft newest latent
        if not ks:
            raise RuntimeError("packed window is empty")
        if fg is not None:
            self._forget_frames = frames_rec
        if soft_bias is not None:
            # the per-key bias of the attended window, or None when every frame's bias is 0 (rho = 1, or nothing aged):
            # the caller then runs the plain attention kernel on the unchanged window (bitwise the default read)
            self._pack_bias = None if all(b == 0.0 for _, b in soft_bias) else torch.cat(
                [torch.full((n,), b, dtype=torch.float32, device=rk.device) for n, b in soft_bias])
        if loc_on:
            self._pack_pos = torch.cat(poss)
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1), src_off, kept

    @staticmethod
    def _forget_split(B, n_m, n_r, x, fg):
        """ equal-budget solver. Allocate a per-frame budget B (< n_m + n_r) between the material and
        rigid tokens so that K_m + K_r = B EXACTLY: proportional to w_role = n_role * rho_role0^x (the nominal per-
        latent keep rates rho_m0, rho_r0; x = age in latents), i.e. K_role = B * w_role / (w_m + w_r) - equivalently
        a common factor c^x = B / (w_m + w_r) applied to both nominal rates, so the solved rates keep the nominal
        ratio rho_m / rho_r = rho_m0 / rho_r0; round K_m, give K_r = B - K_m; if a role would need more tokens than
        it has, cap it at its count and give the remainder to the other role (two roles: the remainder always fits,
        since B < n_m + n_r)."""
        wm = n_m * float(fg["rho_material"]) ** x
        wr = n_r * float(fg["rho_rigid"]) ** x
        km = int(round(B * wm / (wm + wr))) if wm + wr > 0 else min(n_m, B)
        km = max(0, min(km, n_m, B))
        kr = B - km
        if kr > n_r:
            kr = n_r
            km = B - kr
        return km, kr

    @staticmethod
    def _source_redundancy(vj, src_v):
        """: redundancy of every token of a history frame with the source, [tokens]: mean over heads of
        the cosine between the frame's value and the source frame's value at the SAME grid position. Values are used
        because they are never roped (the cache stores them raw in every slot, the source's included), so neither the
        temporal RoPE rotation (which differs per frame) nor slot 0's roped-at-0 key enters the comparison, and V is what
        attention retrieves. Cost per frame: tokens x heads x head_dim multiply-adds (2.4 M at 1560 x 12 x 128)."""
        ct = torch.promote_types(vj.dtype, torch.float32)               # bf16 -> fp32; fp32 / fp64 kept
        a, b = vj[0].to(ct), src_v[0].to(ct)                             # [tokens, heads, head_dim]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        return cos.mean(-1)

    @classmethod
    def _srcred_select(cls, kj, vj, src_v, n, mode):
        """The n tokens of one frame that are LEAST (mode least) or MOST (most) redundant with the source; ties broken
        by larger key norm (stable sorts: by key norm descending first, then by redundancy). Returns (sorted indices,
        redundancy [tokens])."""
        red = cls._source_redundancy(vj, src_v)
        norms = kj[0].float().flatten(1).norm(dim=-1)
        by_norm = torch.sort(-norms, stable=True).indices
        key = red[by_norm] if mode == "least" else -red[by_norm]
        order = by_norm[torch.sort(key, stable=True).indices]
        return order[:n].sort().values, red

    @classmethod
    def _srcred_order(cls, kj, vj, src_v, mode):
        """: the full token ranking of `_srcred_select` (its first n entries are that function's set).
        Returns (ranked indices, redundancy [tokens])."""
        red = cls._source_redundancy(vj, src_v)
        norms = kj[0].float().flatten(1).norm(dim=-1)
        by_norm = torch.sort(-norms, stable=True).indices
        key = red[by_norm] if mode == "least" else -red[by_norm]
        return by_norm[torch.sort(key, stable=True).indices], red

    @staticmethod
    def _stratified_pick(order, fs, n):
        """: n token indices of a frame of fs tokens, one per raster span [(i fs) / n, ((i + 1) fs) / n): in each
        span the token ranked first by `order` (a full ranking, best first); sorted."""
        dev = order.device
        rank = torch.empty(fs, dtype=torch.long, device=dev)
        rank[order] = torch.arange(fs, device=dev)
        cell = (torch.arange(fs, device=dev) * n) // fs
        best = torch.full((n,), fs, dtype=torch.long, device=dev).scatter_reduce(0, cell, rank, reduce="amin")
        return order[best].sort().values

    @staticmethod
    def _forget_select(kj, cand, n, rule):
        """n token indices out of the candidate indices `cand` of one frame: the n largest key norms (keynorm) or a
        fixed even stride over the candidates (stride); sorted."""
        if n <= 0 or cand.numel() == 0:
            return cand[:0]
        if n >= cand.numel():
            return cand
        if rule == "stride":
            return cand[(torch.arange(n, device=cand.device) * cand.numel()) // n]
        norms = kj[0].index_select(0, cand).float().flatten(1).norm(dim=-1)
        return cand[norms.topk(n).indices].sort().values

    # --- related KV-memory baselines: training-free PORTS (opt-in; the default path never reaches these: the
    # attributes `forcing_kv` / `future_kv` / `tether_cache` are absent) --------------------------------------------------
    #   forcing_kv   Forcing-KV (arXiv 2605.09681; official code github.com/zju-jiyicheng/Forcing-KV @50ac1f4, Apache-2.0),
    #                the RELEASED code's behaviour (configs/forcing-kv/forcingkv_causal_forcing_inference.yaml): an offline
    #                per-layer head split (static / dynamic; the shipped Causal-Forcing profile, configs/factorial/
    #                forcingkv_heads_causalforcing.json, byte-identical to their Self-Forcing profile) and per head class:
    #                  static heads  : [sink, transition anchor, current block]  (anchor = the newest history latent)
    #                  dynamic heads : [sink, dynamic buffer, anchor, current block]
    #                  layer 0       : every head as static (`use_grouped_kv` requires layer_idx != 0)
    #                dynamic buffer: the `candidates` (3) latents before the anchor, each cut into `segments` (6) contiguous
    #                raster bands; band score = mean over its tokens of cos(K_f, K_{f+1}) flattened over the score layer's
    #                DYNAMIC heads x head_dim (post-RoPE keys; f+1 = the next latent, the anchor for the last candidate);
    #                keep the round(3 * 6 * retention 0.33) = 6 LEAST similar bands jointly over the candidates (topk
    #                largest=False, sorted); computed at layer `score_layer` (1) and shared by every later layer; rebuilt
    #                every block (the released buffer is overwritten each chunk, not accumulated -.
    #                Sink = the source (sink_size 1 in the released config). Read-time selection from the host window
    #                (the cache is never written); kept tokens keep their window RoPE. In the released code the selection
    #                is made on the cache-writing clean pass of the previous chunk; here it is recomputed from the same
    #                committed clean keys at every evaluation of the block (identical inputs, identical selection).
    def _fkv_attention(self, rq, rk, vv, seq, cur, frame_seqlen):
        """Forcing-KV read for one block. Returns (out, k_static, v_static, kept_static, dynamic_tokens)."""
        cfg, fs = self.forcing_kv, frame_seqlen
        pos = {}
        for j, f in enumerate(seq):
            if f is not None and f not in pos:
                pos[f] = j
        if 0 not in pos:
            raise RuntimeError("forcing_kv: the source is not in the attended window")
        cur_slots = [j for j, f in enumerate(seq) if f is not None and f >= cur]
        hist = [f for f in pos if 0 < f < cur]
        anchor = max(hist) if hist else None
        C, n = int(cfg["candidates"]), int(cfg["segments"])
        cands = [] if anchor is None else [f for f in range(anchor - C, anchor) if f >= 1 and f in pos and f + 1 in pos]
        bounds = [(i * fs) // n for i in range(n + 1)]
        nkeep = int(round(len(cands) * n * float(cfg["retention"])))

        def sl(t, j):
            return t[:, j * fs:(j + 1) * fs]
        static = [pos[0]] + ([pos[anchor]] if anchor is not None else []) + cur_slots
        k_st = torch.cat([sl(rk, j) for j in static], dim=1)
        v_st = torch.cat([sl(vv, j) for j in static], dim=1)
        kept = [(seq[j], fs) for j in static]
        layer, dyn, sh = int(cfg["layer"]), cfg["dynamic_heads"], cfg["shared"]
        if layer == int(cfg["score_layer"]):
            keep = []
            if nkeep > 0:
                dh = dyn.to(rk.device)
                scores = []
                for f in cands:
                    a = sl(rk, pos[f])[0].index_select(1, dh).flatten(1).float()
                    b = sl(rk, pos[f + 1])[0].index_select(1, dh).flatten(1).float()
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                    scores.extend(cos[bounds[i]:bounds[i + 1]].mean() for i in range(n))
                flat = torch.stack(scores)
                keep = torch.topk(flat, k=nkeep, largest=False, dim=-1).indices.sort().values.tolist()
                sh["scores"] = flat.detach()
            sh.update(cur=cur, cands=list(cands), keep=keep)
        if layer == 0 or dyn is None or dyn.numel() == 0:
            return attention(rq, k_st, v_st), k_st, v_st, kept, 0
        if sh.get("cur") != cur or sh.get("cands") != cands:
            raise RuntimeError("forcing_kv: no shared band selection for this block (the score layer runs first)")
        parts = [torch.arange(pos[cands[c // n]] * fs + bounds[c % n], pos[cands[c // n]] * fs + bounds[c % n + 1])
                 for c in sh["keep"]]
        lead_k, lead_v = sl(rk, pos[0]), sl(vv, pos[0])
        if not parts:                       # empty dynamic buffer: every head attends the static set
            return attention(rq, k_st, v_st), k_st, v_st, kept, 0
        if parts:
            di = torch.cat(parts).to(rk.device)
            k_dy = torch.cat([lead_k, rk.index_select(1, di), k_st[:, fs:]], dim=1)
            v_dy = torch.cat([lead_v, vv.index_select(1, di), v_st[:, fs:]], dim=1)
            n_dyn = int(di.numel())
        else:
            k_dy, v_dy, n_dyn = k_st, v_st, 0
        dh = dyn.to(rq.device)
        o_dy = attention(rq.index_select(2, dh), k_dy.index_select(2, dh), v_dy.index_select(2, dh))
        out = torch.empty(rq.shape, dtype=o_dy.dtype, device=rq.device)
        out.index_copy_(2, dh, o_dy)
        sth = cfg["static_heads"]
        if sth is not None and sth.numel():
            sth = sth.to(rq.device)
            out.index_copy_(2, sth, attention(rq.index_select(2, sth), k_st.index_select(2, sth), v_st.index_select(2, sth)))
        return out, k_st, v_st, kept, n_dyn

    #   future_kv    Future Forcing (arXiv 2605.30083; no code released): future-aware KV eviction + redundancy-aware merging,
    #                with the paper's lambda 0.5, Delta_max 6, tau 0.95, at OUR equal attended-token budget (per block the
    #                sum of the fg_uni / fg_rho03 per-frame budgets over the window's history frames; the paper's own budget
    #                of 18 latent frames is >= our 21-frame window's history, i.e. a no-op here). Per layer, once per block
    #                (first evaluation), over the ALIVE history tokens of the window (persistent state: an evicted token is
    #                never attended again; merged values persist and compound):
    #                  future term  Qbar = mean of the pre-RoPE queries of the last `w` committed latents (clean refresh
    #                               pass); proxy Q~_D = RoPE(Qbar) at temporal position (last query position + D), each token
    #                               at its own (h, w); a^_D[h, j] = sum over the fs proxy rows of softmax over [source + alive
    #                               history] keys; future_j = mean over D and heads
    #                  history term acc_j / cnt_j: at every clean refresh pass the committed latents' attention on each
    #                               attended key (sum over a latent's queries, mean over heads), averaged over the latents
    #                               that attended it ("average historical attention across frames")
    #                  score_j = lambda future_j + (1 - lambda) history_j; keep the top-B (ties: stable, older first)
    #                  merge: profile_j = [K_j . mean_rows(Q~_D)_h] over (h, D); every evicted j joins its most cosine-
    #                               similar retained i if cos >= tau, else dropped; V~_i = mean_D (a^_D,i V_i + sum_j a^_D,j
    #                               V_j) / (a^_D,i + sum_j a^_D,j) per head; K_i unchanged (paper Eq. 3).
    #                The merged values live in the port's own per-layer state, NOT in the host KV cache / Echo memory (the host
    #                memory still decides which frames exist; a frame that leaves the host window is forgotten by the port).
    #                Choices where the paper is silent: source kept whole (as every arm), current block whole,
    #                w = 3, mean over heads and one selection per layer, eviction once per block, no RoPE remap (kept tokens
    #                keep their window positions), proxy rows pooled per head for the merge profile.
    @staticmethod
    def _ffk_mass(q, k, chunk=512):
        """sum over the query rows of softmax(q k^T / sqrt(d)) per head: q [Lq, n, d], k [Lk, n, d] -> [n, Lk] (float32)."""
        q, k = q.float(), k.float()
        out = torch.zeros(k.shape[1], k.shape[0], dtype=torch.float32, device=k.device)
        for s in range(0, q.shape[0], chunk):
            lg = torch.einsum("qnd,knd->nqk", q[s:s + chunk], k) / math.sqrt(q.shape[-1])
            out += lg.softmax(-1).sum(1)
        return out

    @staticmethod
    def _ffk_merge(V, a, S, E, prof, tau):
        """Future Forcing Eq. 3. V [L, n, d] float32 candidate values, a [D, n, L] future weights, S / E retained / evicted
        candidate indices, prof [L, P] merge profiles. Returns (V with the merged retained rows replaced, number of evicted
        tokens merged, number of retained tokens that received a merge)."""
        if E.numel() == 0 or S.numel() == 0:
            return V, 0, 0
        pn = torch.nn.functional.normalize(prof.float(), dim=-1)
        ps = pn.index_select(0, S)
        best, arg = [], []
        for s in range(0, E.numel(), 4096):
            sim = pn.index_select(0, E[s:s + 4096]) @ ps.T
            b, i = sim.max(dim=1)
            best.append(b)
            arg.append(i)
        best, arg = torch.cat(best), torch.cat(arg)
        ok = best >= tau
        if not bool(ok.any()):
            return V, 0, 0
        ej, tgt = E[ok], arg[ok]
        VS, VE = V.index_select(0, S), V.index_select(0, ej)
        acc = torch.zeros_like(VS)
        for D in range(a.shape[0]):
            aS = a[D].index_select(1, S).T                         # [|S|, n]
            aE = a[D].index_select(1, ej).T                        # [|E'|, n]
            num = aS[..., None] * VS
            den = aS.clone()
            num.index_add_(0, tgt, aE[..., None] * VE)
            den.index_add_(0, tgt, aE)
            acc += num / den.clamp_min(1e-30)[..., None]
        merged = torch.zeros(S.numel(), dtype=torch.bool, device=S.device)
        merged[tgt] = True
        V = V.clone()
        V[S[merged]] = (acc / a.shape[0])[merged]
        return V, int(ej.numel()), int(merged.sum())

    def _ffk_attention(self, q, rq, rk, vv, seq, cur, frame_seqlen, grid_sizes, freqs, qstart):
        """Future Forcing read (eviction once per block) + the refresh-pass bookkeeping. Returns (out, k_view, v_view, kept)."""
        cfg, st, fs = self.future_kv, self._ff_state, frame_seqlen
        nq = rq.shape[1] // fs
        W = self.max_attention_size // fs
        pos = {}
        for j, f in enumerate(seq):
            if f is not None and f not in pos:
                pos[f] = j
        if 0 not in pos:
            raise RuntimeError("future_kv: the source is not in the attended window")
        hist = sorted(f for f in pos if 0 < f < cur)
        newest, R, rho = cur + nq - 1, int(cfg["recent_latents"]), float(cfg["rho"])
        B = sum(fs if f > newest - R else int(fs * rho ** min(cur - f, W - 1) + 0.5) for f in hist)

        def sl(t, j):
            return t[:, j * fs:(j + 1) * fs]
        alive = st["alive"]
        if st["cur"] != cur:
            st["cur"] = cur
            for f in list(alive):
                if f not in pos:
                    del alive[f]                       # left the host window: forgotten (re-enters whole if it returns)
            idxs = [alive[f][0] if f in alive else torch.arange(fs, device=rk.device) for f in hist]
            total = sum(int(i.numel()) for i in idxs)
            info = dict(block=int(cur), budget=int(B), candidates=int(total), kept=int(min(total, B)), merged=0,
                        merged_into=0)
            if total > B:
                Kc = torch.cat([sl(rk, pos[f]).index_select(1, i) for f, i in zip(hist, idxs)], dim=1)[0]
                Vc = torch.cat([alive[f][1] if (f in alive and alive[f][1] is not None) else sl(vv, pos[f]).index_select(1, i)
                                for f, i in zip(hist, idxs)], dim=1)[0]
                if not st["qbuf"]:
                    raise RuntimeError("future_kv: no committed queries for the future proxy")
                qbar = torch.stack([qq for _, qq in st["qbuf"]]).float().mean(0)           # [1, fs, n, d]
                one = grid_sizes.clone()
                one[0][0] = 1
                P = int(qstart) + nq - 1
                kn = torch.cat([sl(rk, pos[0])[0], Kc], dim=0)
                a, prof = [], []
                for D in range(1, int(cfg["horizon"]) + 1):
                    qt = block_relativistic_rope(qbar, one, freqs, P + D, theta_by_head=self.rope_theta_by_head)[0]
                    a.append(self._ffk_mass(qt, kn)[:, fs:])                                # [n, Lc]
                    prof.append(torch.einsum("knd,nd->kn", Kc.float(), qt.float().mean(0)))  # [Lc, n]
                a = torch.stack(a)                                                          # [D, n, Lc]
                fut = a.mean(dim=(0, 1))
                hterm = []
                for f, i in zip(hist, idxs):
                    acc, cnt = st["acc"].get(f), st["cnt"].get(f)
                    if acc is None:
                        hterm.append(torch.zeros(int(i.numel()), device=rk.device))
                    else:
                        c = cnt.index_select(0, i)
                        hterm.append(torch.where(c > 0, acc.index_select(0, i) / c.clamp_min(1), torch.zeros_like(c)))
                hterm = torch.cat(hterm)
                lam = float(cfg["lam"])
                score = lam * fut + (1.0 - lam) * hterm
                order = torch.sort(score, descending=True, stable=True).indices
                S, E = order[:B].sort().values, order[B:].sort().values
                V2, nm, ni = Vc.float(), 0, 0
                if cfg["merge"]:
                    V2, nm, ni = self._ffk_merge(V2, a, S, E, torch.stack(prof, dim=-1).flatten(1), float(cfg["tau"]))
                keep = torch.zeros(Kc.shape[0], dtype=torch.bool, device=Kc.device)
                keep[S] = True
                V2 = V2.to(vv.dtype)
                o = 0
                for f, i in zip(hist, idxs):
                    m = int(i.numel())
                    sel = keep[o:o + m]
                    alive[f] = (i[sel], V2[o:o + m][sel][None])
                    o += m
                info.update(merged=nm, merged_into=ni, score_kept_mean=float(score[S].mean()) if B > 0 else None,
                            score_evicted_mean=float(score[E].mean()) if E.numel() else None)
            st["last"] = info
        ks, vs, kept, owners = [], [], [], []
        off, done = 0, set()
        for j, f in enumerate(seq):
            if f is None or f in done:
                continue                                   # compressed / blended frames carry no id: not tracked, dropped
            done.add(f)
            if f == 0 or f >= cur or f not in alive:
                kj, vj = sl(rk, j), sl(vv, j)
                idx = None
            else:
                idx, v_al = alive[f]
                kj = sl(rk, j).index_select(1, idx)
                vj = v_al if v_al is not None else sl(vv, j).index_select(1, idx)
            ks.append(kj)
            vs.append(vj)
            kept.append((f, int(kj.shape[1])))
            owners.append((f, idx, off, int(kj.shape[1])))
            off += int(kj.shape[1])
        k_view, v_view = torch.cat(ks, dim=1), torch.cat(vs, dim=1)
        out = attention(rq, k_view, v_view)
        if _DENOISE_STEP_FRACTION is None and not _BRANCH_READONLY:
            # clean refresh pass: history attention of the committed latents and the query buffer of the future proxy
            tot = torch.zeros(k_view.shape[1], dtype=torch.float32, device=k_view.device)
            for s in range(nq):
                tot += self._ffk_mass(rq[0, s * fs:(s + 1) * fs], k_view[0]).mean(0)
            for f, idx, o, m in owners:
                if f == 0:
                    continue
                if f not in st["acc"]:
                    st["acc"][f] = torch.zeros(fs, dtype=torch.float32, device=k_view.device)
                    st["cnt"][f] = torch.zeros(fs, dtype=torch.float32, device=k_view.device)
                ii = torch.arange(fs, device=k_view.device) if idx is None else idx
                st["acc"][f].index_add_(0, ii, tot[o:o + m])
                st["cnt"][f].index_add_(0, ii, torch.full((m,), float(nq), device=k_view.device))
            for s in range(nq):
                st["qbuf"].append((cur + s, q[:, s * fs:(s + 1) * fs].detach().float().clone()))
            st["qbuf"] = st["qbuf"][-int(cfg["w"]):]
        return out, k_view, v_view, kept

    #   tether_cache TetherCache (arXiv 2606.13035; official code github.com/my4f175/TetherCache @37c581a, Apache-2.0) on the
    #                static sink (its native base is Self-Forcing's sink 3 + FIFO window): a 21-frame read window = sink S (3:
    #                the source + latents 1, 2) | memory M (14 latents, the port's OWN per-layer store, chronological slots) |
    #                recent R (4 latents INCLUDING the current block, so R - k history latents from the host buffer) - the same
    #                21 frames as the host, re-roped by slot (keys stored unrotated, block-relative RoPE on read, as released).
    #                Warm-up = the host path, bitwise; at the clean refresh pass of the last block before the host window would
    #                overflow the memory is seeded with latents S .. S+M-1 (no scoring, no TAME). Afterwards, at every clean
    #                refresh pass, the latents that have left the recent window since the last admission compete with the
    #                memory (whole latents, per layer):
    #                  GRAB  l(c) = mean over heads, the block's clean pre-RoPE queries and c's unrotated keys of q.k / sqrt(d)
    #                        (= <mean q, mean k_c> per head, the exact identity of the released full product); phi_imp =
    #                        softmax over the pool; phi_div = max(0, 1 - max_{c' != c} exp(-|g_c - g_c'| / sigma) phi_imp(c')),
    #                        sigma = max(1, (max g - min g + 1) / 2), g = latent index; keep the top M of phi_imp + alpha
    #                        phi_div (alpha 0.35), kept members in slot order then the admitted latents
    #                  TAME  every NEWLY admitted latent: x <- (1 - tau) x + tau (sigma_T (x - mu_x) / sigma_x + mu_T) for K
    #                        (unrotated) and V separately, statistics per head and channel over tokens, trusted pool = sink +
    #                        the memory before the rewrite (tau 0.6, the paper's default; the released code defaults 0.35)
    #                While a block is denoised, the latents that just left the recent window are attended nowhere (as released);
    #                the refresh pass attends the updated memory. The admitted / edited K/V are WRITTEN into the port's memory
    #                store (the method's own cache region); the host KV buffer is never written.
    def _tc_admit(self, q, raw_k, vwin, pos, cands, frame_seqlen, grid_sizes, freqs):
        cfg, st, fs = self.tether_cache, self._tc_state, frame_seqlen
        M = int(cfg["memory"])
        mk, mv, mids = st["mem_k"], st["mem_v"], list(st["mem_ids"])
        n, d = mk.shape[2], mk.shape[3]
        ck = torch.cat([raw_k[:, pos[f] * fs:(pos[f] + 1) * fs] for f in cands], dim=1)
        cv = torch.cat([vwin[:, pos[f] * fs:(pos[f] + 1) * fs] for f in cands], dim=1)
        pk = torch.cat([mk, ck], dim=1)[0].reshape(-1, fs, n, d)
        pv = torch.cat([mv, cv], dim=1)[0].reshape(-1, fs, n, d)
        g_ids = mids + list(cands)
        P = len(g_ids)
        info = dict(block=None, pool=g_ids, admitted=[], demoted=[])
        if P <= M:
            sel = list(range(P))
        else:
            qm = q[0].float().mean(0)                                             # [n, d]
            ell = (pk.float().mean(1) * qm).sum(-1).mean(-1) / math.sqrt(d)       # [P]
            imp = torch.softmax(ell / float(cfg["temperature"]), dim=-1)
            g = torch.tensor(g_ids, dtype=torch.float32, device=imp.device)
            sigma = max(1.0, 0.5 * float(g.max() - g.min() + 1))
            sim = torch.exp(-(g[:, None] - g[None, :]).abs() / sigma) * imp[None, :]
            sim.fill_diagonal_(0.0)
            div = (1.0 - sim.max(dim=-1).values).clamp_min(0.0)
            score = imp + float(cfg["alpha"]) * div
            top = torch.topk(score, M).indices
            mask = torch.zeros(P, dtype=torch.bool, device=score.device)
            mask[top] = True
            sel = mask.nonzero(as_tuple=True)[0].tolist()
            info.update(imp=[float(x) for x in imp], score=[float(x) for x in score])
        new = [p for p in sel if p >= len(mids)]
        out_k, out_v = [], []
        if new:
            # trusted pool = sink (slot 0's stored key is roped at frame 0: un-rotated here) + the memory before the rewrite
            s = int(self.sink_size)
            sk = raw_k[:, :s * fs].clone()
            if pos.get(0) != 0:
                raise RuntimeError("tether_cache: the source is not in slot 0")
            one = grid_sizes.clone()
            one[0][0] = 1
            sk[:, :fs] = block_relativistic_rope(raw_k[:, :fs], one, freqs.conj(), 0).type_as(raw_k)   # inverse RoPE
            tk = torch.cat([sk, mk], dim=1).float()
            tv = torch.cat([vwin[:, :s * fs], mv], dim=1).float()
            stats = [(t.mean(dim=(0, 1)), t.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)) for t in (tk, tv)]
        tau = float(cfg["tau"])
        for p in sel:
            fk, fv = pk[p][None], pv[p][None]
            if p >= len(mids):
                edited = []
                for x, (mu_t, sd_t) in zip((fk, fv), stats):
                    x32 = x.float()
                    mu, sd = x32.mean(dim=(0, 1)), x32.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
                    edited.append(((1.0 - tau) * x32 + tau * (sd_t * (x32 - mu) / sd + mu_t)).to(x.dtype))
                fk, fv = edited
            out_k.append(fk)
            out_v.append(fv)
        st["mem_k"], st["mem_v"] = torch.cat(out_k, dim=1), torch.cat(out_v, dim=1)
        st["mem_ids"] = [g_ids[p] for p in sel]
        info.update(admitted=[g_ids[p] for p in new], demoted=[f for f in mids if f not in st["mem_ids"]],
                    memory=list(st["mem_ids"]))
        return info

    def _tc_attention(self, q, raw_k, vwin, seq, cur, frame_seqlen, grid_sizes, freqs):
        """TetherCache read on the static sink. Returns None while the host path applies (warm-up), else
        (out, roped window keys, values, kept)."""
        cfg, st, fs = self.tether_cache, self._tc_state, frame_seqlen
        nq = q.shape[1] // fs
        W = self.max_attention_size // fs
        S, M, R = int(self.sink_size), int(cfg["memory"]), int(cfg["recent"])
        pos = {}
        for j, f in enumerate(seq):
            if f is not None and f not in pos:
                pos[f] = j
        commit = _DENOISE_STEP_FRACTION is None and not _BRANCH_READONLY
        if not st["seeded"]:
            if commit and cur + 2 * nq > W:
                ids = list(range(S, S + M))
                if any(f not in pos for f in ids):
                    raise RuntimeError("tether_cache: the seed latents are not in the host window")
                st["mem_k"] = torch.cat([raw_k[:, pos[f] * fs:(pos[f] + 1) * fs] for f in ids], dim=1).clone()
                st["mem_v"] = torch.cat([vwin[:, pos[f] * fs:(pos[f] + 1) * fs] for f in ids], dim=1).clone()
                st["mem_ids"], st["next"], st["seeded"] = ids, S + M, True
                st["events"].append(dict(block=int(cur), seeded=ids))
            return None
        first_recent = cur - (R - nq)
        if commit:
            cands = list(range(int(st["next"]), first_recent))
            if cands:
                if any(f not in pos for f in cands):
                    raise RuntimeError("tether_cache: an evicted latent is no longer in the host window")
                info = self._tc_admit(q, raw_k, vwin, pos, cands, fs, grid_sizes, freqs)
                info["block"] = int(cur)
                st["events"].append(info)
                st["next"] = first_recent
        ks = [raw_k[:, :S * fs], st["mem_k"]]
        vs = [vwin[:, :S * fs], st["mem_v"]]
        ids = list(range(S)) + list(st["mem_ids"])
        for f in list(range(first_recent, cur)) + list(range(cur, cur + nq)):
            if f not in pos:
                raise RuntimeError(f"tether_cache: latent {f} is not in the host window")
            ks.append(raw_k[:, pos[f] * fs:(pos[f] + 1) * fs])
            vs.append(vwin[:, pos[f] * fs:(pos[f] + 1) * fs])
            ids.append(f)
        k_raw, v_all = torch.cat(ks, dim=1), torch.cat(vs, dim=1)
        F = k_raw.shape[1] // fs
        grid = grid_sizes.clone()
        grid[0][0] = F
        rk = block_relativistic_rope(k_raw, grid, freqs, 0, theta_by_head=self.rope_theta_by_head).type_as(v_all)
        rk[:, :fs] = raw_k[:, :fs]                       # slot 0: the stored (roped-at-0) source key, as on the host path
        rq = block_relativistic_rope(q, grid_sizes, freqs, F - nq, theta_by_head=self.rope_theta_by_head).type_as(v_all)
        return attention(rq, rk, v_all), rq, rk, [(f, fs) for f in ids]

    def _port_record(self, port, kept, frame_seqlen):
        """: per-block row of the related-KV ports (layer 0, first evaluation; logging only). For Forcing-KV the
        layer-0 row is the static set; `head_mean_tokens` adds the dynamic buffer weighted by the share of (layer, head)
        pairs that attend it (layer 0 counted static, as released)."""
        row = self.context_log[-1]
        if port["name"] == "forcing_kv":
            cfg = self.forcing_kv
            ids = [f for f, _ in kept]
            hist = [f for f in ids if f not in (0,) and f < row["block_start"]]
            anchor = max(hist) if hist else None
            n = int(cfg["segments"])
            nc = 0 if anchor is None else sum(1 for f in range(anchor - int(cfg["candidates"]), anchor)
                                               if f >= 1 and f in row["latent_ids"] and f + 1 in row["latent_ids"])
            dyn = int(round(nc * n * float(cfg["retention"]))) * frame_seqlen // n
            port = dict(port, static_tokens=row["attended_tokens"], dynamic_buffer_tokens=dyn,
                        dynamic_head_fraction=float(cfg["dynamic_fraction"]),
                        head_mean_tokens=row["attended_tokens"] + float(cfg["dynamic_fraction"]) * dyn)
        row["port"] = port

    def _context_record(self, seq, cur, nq, rq, rk, src_off, frame_seqlen, soft, gamma, kept=None):
        """One row per generated block (first denoising evaluation): attended key frames, their latent
        ids (duplicates = a latent attended twice, e.g. pinlat1 at k = 3; recent_coverage = how many
        of the 21 latents before the block are attended; blended = compressed / EMA frames without an
        id), source attended, which entry the scheduled read uses, and the subsampled source-key
        attention mass: with a C3 map, separately for low-variation queries (they attend the sharp
        window) and high-variation queries (the scheduled window); otherwise over all queries on the
        window as attended. Up to 256 evenly spaced query rows per class."""
        ids = [i for i in seq if i is not None]
        blended = len(seq) - len(ids)
        if kept is None:
            kept = [(fid, frame_seqlen) for fid in seq]
        row = dict(block_start=int(cur), frames=len(seq), latent_ids=sorted(ids), blended=blended,
                   duplicates=len(ids) - len(set(ids)),
                   recent_coverage=len(set(ids) & set(range(max(0, cur - 21), cur))),
                   source_attended=(0 in ids), source_entry=None, mass_support=None, mass_material=None,
                   mass_all=None, tokens_per_frame=[[fid, n] for fid, n in kept],
                   attended_tokens=int(sum(n for _, n in kept)), full_window_tokens=len(seq) * frame_seqlen,
                   frames_kept=len(kept))
        if getattr(self, 'forget', None) is not None and getattr(self, '_forget_frames', None) is not None:
            #: kept tokens per history frame and role vs the uniform budget (equal by construction)
            fr = self._forget_frames
            role = getattr(self, 'forget_role', None)
            row["forget"] = dict(frames=fr, total_kept=int(sum(e["kept"] for e in fr)),
                                 uniform_total=int(sum(e["budget"] for e in fr)),
                                 map_version=int(getattr(self, 'forget_map_version', 0)),
                                 material_fraction=None if role is None else float(role.float().mean()))
            rk_ = [e for e in fr if e.get("red_kept_sum") is not None]
            if rk_:
                #: token-weighted mean source redundancy of the kept vs the dropped history tokens
                nk, nd = sum(e["kept"] for e in rk_), sum(e["n_dropped"] for e in rk_)
                row["forget"]["red_kept_mean"] = sum(e["red_kept_sum"] for e in rk_) / max(1, nk)
                row["forget"]["red_dropped_mean"] = (sum(e["red_dropped_sum"] for e in rk_) / nd) if nd else None
                row["forget"]["reference"] = sorted({str(e.get("reference")) for e in rk_})
        if src_off is not None and 0 in ids and any(fid == 0 for fid, _ in kept):
            read = getattr(self, 'source_read', 'sharp')
            use_soft = soft is not None and read == 'soft'
            row["source_entry"] = "soft" if use_soft else "sharp"
            k_sched = rk
            if use_soft:
                k_sched = torch.cat([rk[:, :src_off], soft[0].type_as(rk), rk[:, src_off + frame_seqlen:]], dim=1)
            role = getattr(self, 'source_role', None)
            Lq = rq.shape[1]

            def sub(mask):
                where = mask.nonzero(as_tuple=True)[0]
                if where.numel() > 256:
                    where = where[torch.linspace(0, where.numel() - 1, 256).long()]
                return where.to(rq.device)
            if role is not None:
                support = role.to(rq.device).repeat(Lq // frame_seqlen)
                row["mass_support"] = self._source_mass(rq, rk, src_off, frame_seqlen, sub(support), gamma)
                row["mass_material"] = self._source_mass(rq, k_sched, src_off, frame_seqlen, sub(~support), gamma)
            else:
                row["mass_all"] = self._source_mass(rq, k_sched, src_off, frame_seqlen,
                                                    sub(torch.ones(Lq, dtype=torch.bool)), gamma)
        self.context_log.append(row)

    def _echo_support(self, old_v, follow_v, frame_seqlen):
        """ H2: temporal support of every evicted token [b, L*fs]: for evicted latent i, s = min(c1, c2) with
        c1 / c2 = the best cosine (raw values, all heads, unit-normalised) between the token and the latents i + 1 / i + 2
        of the committed sequence (evicted entry then the following recent entries) within +-R / +-2R grid tokens.
        Returns (support [b, L*fs] float32 or None per latent without two followers, n_latents_scored)."""
        adm = self.echo_admission
        h, w = self._echo_hw
        R = int(adm["radius"])
        seq = old_v if follow_v is None else torch.cat([old_v, follow_v], dim=1)
        b, F = seq.shape[0], seq.shape[1] // frame_seqlen
        L = old_v.shape[1] // frame_seqlen
        x = seq.float().reshape(b, F, h, w, -1)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out = torch.full((b, L, h, w), float("nan"), device=seq.device)

        def best(a, c, r):
            cp = torch.nn.functional.pad(c.permute(0, 3, 1, 2), (r, r, r, r)).permute(0, 2, 3, 1)  # zero pad -> cos 0
            m = None
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    cs = (a * cp[:, dy:dy + h, dx:dx + w]).sum(-1)
                    m = cs if m is None else torch.maximum(m, cs)
            return m
        n = 0
        for i in range(L):
            if i + 2 >= F:
                continue                                    # not enough committed followers: abstain
            out[:, i] = torch.minimum(best(x[:, i], x[:, i + 1], R), best(x[:, i], x[:, i + 2], 2 * R))
            n += 1
        return out.reshape(b, L * frame_seqlen), n

    def _echo_admission_adj(self, old_v, follow_v, frame_seqlen):
        """ H2: per evicted token adjustment unit in [-1, 0]: 0 (abstain, baseline score) unless the token's
        temporal support is a robust low outlier of its own latent (z = (median - s) / (1.4826 MAD) > kappa); linear to
        -1 at z = 2 kappa. Tokens without two committed followers abstain."""
        sup, n = self._echo_support(old_v, follow_v, frame_seqlen)
        kap = float(self.echo_admission["kappa"])
        b, L = sup.shape[0], sup.shape[1] // frame_seqlen
        s = sup.reshape(b, L, frame_seqlen)
        med = s.nanmedian(dim=-1, keepdim=True).values
        mad = (s - med).abs().nanmedian(dim=-1, keepdim=True).values * 1.4826 + 1e-6
        z = (med - s) / mad
        unit = -((z - kap) / kap).clamp(0.0, 1.0)
        unit = torch.nan_to_num(unit, nan=0.0).reshape(b, L * frame_seqlen)
        return unit, n

    def _echo_compress(self, old_k, old_v, frame_seqlen, old_frame_index=None, follow_v=None):
        """Bounded calibrated-query compression of evicted recent tokens."""
        if self.echo_compressed_frames <= 0:
            return
        adm = getattr(self, 'echo_admission', None)
        if adm is not None:
            new_adj, n_scored = self._echo_admission_adj(old_v, follow_v, frame_seqlen)
            cand_adj = new_adj if self._echo_compressed_k is None or self._echo_compressed_adj is None else torch.cat(
                [self._echo_compressed_adj, new_adj], dim=1)
        cand_k = old_k if self._echo_compressed_k is None else torch.cat(
            [self._echo_compressed_k, old_k], dim=1)
        cand_v = old_v if self._echo_compressed_v is None else torch.cat(
            [self._echo_compressed_v, old_v], dim=1)
        preserve = getattr(self, 'echo_preserve_spatial', False)
        material = getattr(self, 'material_reader', None) is not None
        if material:
            if old_frame_index is None:
                raise RuntimeError('Material cache requires actual source timestamps')
            origin = (torch.arange(old_k.shape[1], device=old_k.device) +
                      old_frame_index * frame_seqlen)[None].expand(old_k.shape[0], -1)
            cand_origin = (origin if self._echo_compressed_k is None else
                           torch.cat([self._echo_compressed_origin, origin], dim=1))
        if preserve:
            old_spatial = torch.arange(old_k.shape[1], device=old_k.device).remainder(frame_seqlen)
            old_spatial = old_spatial[None, :].expand(old_k.shape[0], -1)
            cand_spatial = (old_spatial if self._echo_compressed_k is None else
                            torch.cat([self._echo_compressed_spatial, old_spatial], dim=1))
        budget = self.echo_compressed_frames * frame_seqlen
        if cand_k.shape[1] <= budget:
            self._echo_compressed_k, self._echo_compressed_v = cand_k, cand_v
            if material:
                self._echo_compressed_origin = cand_origin
            if preserve:
                self._echo_compressed_spatial = cand_spatial
            if adm is not None:
                self._echo_compressed_adj = cand_adj
            return
        if self._echo_q_center is None:
            score = cand_k.float().square().mean(dim=(-1, -2))
        else:
            score = (cand_k.float() * self._echo_q_center.float()).mean(dim=(-1, -2))
        if adm is not None:
            # bounded: at most lam standard deviations of this candidate pool's baseline score
            base_keep = score.topk(budget, dim=1, sorted=True).indices
            score = score + cand_adj * float(adm["lam"]) * score.std(dim=1, keepdim=True)
        keep = score.topk(budget, dim=1, sorted=True).indices
        if adm is not None:
            self._echo_compressed_adj = cand_adj.gather(1, keep)
            st = getattr(self, 'echo_admission_log', None)
            if st is not None:
                changed = budget - int(torch.isin(keep[0], base_keep[0]).sum())
                st.append(dict(evicted=None if old_frame_index is None else int(old_frame_index),
                               latents_scored=int(n_scored), new_tokens=int(new_adj.shape[1]),
                               penalised=int((new_adj[0] < 0).sum()), full_penalty=int((new_adj[0] <= -1).sum()),
                               kept_changed=int(changed), budget=int(budget)))
                e = st[-1]
                print(f"[echo_admission] evicted {e['evicted']}: scored {e['latents_scored']} latents, penalised "
                      f"{e['penalised']}/{e['new_tokens']} (full {e['full_penalty']}), kept set changed {e['kept_changed']}/{budget}")
        gather = keep[:, :, None, None].expand(-1, -1, cand_k.shape[2], cand_k.shape[3])
        self._echo_compressed_k = cand_k.gather(1, gather)
        self._echo_compressed_v = cand_v.gather(1, gather)
        if material:
            self._echo_compressed_origin = cand_origin.gather(1, keep)
        if preserve:
            self._echo_compressed_spatial = cand_spatial.gather(1, keep)

    def _echo_commit(self, q, k, v, frame_index, frame_seqlen):
        q_center = q.detach().float().mean(dim=1, keepdim=True)
        self._echo_q_count += 1
        if self._echo_q_center is None:
            self._echo_q_center = q_center
        elif self._echo_q_count <= self.echo_anchor_pool:
            n = float(self._echo_q_count)
            self._echo_q_center.add_((q_center - self._echo_q_center) / n)

        if frame_index < self.echo_anchor_pool:
            if frame_index in self._echo_early_ids:
                pos = self._echo_early_ids.index(frame_index)
                self._echo_early_k[pos], self._echo_early_v[pos] = k.detach(), v.detach()
            else:
                self._echo_early_ids.append(frame_index)
                self._echo_early_k.append(k.detach())
                self._echo_early_v.append(v.detach())

        if self._echo_recent_ids and self._echo_recent_ids[-1] == frame_index:
            self._echo_recent_k[-1], self._echo_recent_v[-1] = k.detach(), v.detach()
        else:
            self._echo_recent_ids.append(frame_index)
            self._echo_recent_k.append(k.detach())
            self._echo_recent_v.append(v.detach())
        while len(self._echo_recent_k) > self.echo_recent_past:
            old_k, old_v = self._echo_recent_k.pop(0), self._echo_recent_v.pop(0)
            old_id = self._echo_recent_ids.pop(0)
            if frame_index >= self.echo_anchor_pool:
                follow = None
                if getattr(self, 'echo_admission', None) is not None:
                    # H2: the committed clean latents that FOLLOW the evicted entry (still in recent memory)
                    follow = torch.cat(self._echo_recent_v, dim=1) if self._echo_recent_v else None
                self._echo_compress(old_k, old_v, frame_seqlen, old_id, follow_v=follow)

    def _echo_cadence_push(self, q, k, v, frame_index, group_start, frame_seqlen):
        """factorial.echo_cadence_latents, nfpb 1 only). Generated latents are
        grouped in runs of `cad` from latent 1 ([1..cad], [cad+1..2cad], ...). The clean K/V of
        each latent is buffered; when the group is complete its K/V are concatenated into ONE
        Echo entry keyed by the group start (the entry shape of a chunk-`cad` deployment) and
        committed through the unchanged `_echo_commit`, so commit, compression and the q-center
        update fire once per group. Anchor rotation is keyed by the group start in
        `_echo_cached_attention`. Pending latents stay visible to later queries as plain parts."""
        cad = int(self.echo_cadence_latents)
        if frame_index in self._echo_pending_ids:            # re-refresh of the same latent
            pos = self._echo_pending_ids.index(frame_index)
            self._echo_pending_q[pos] = q.detach()
            self._echo_pending_k[pos], self._echo_pending_v[pos] = k.detach(), v.detach()
        else:
            expected = group_start + len(self._echo_pending_ids)
            if self._echo_pending_ids and (self._echo_pending_ids[0] != group_start
                                           or frame_index != expected):
                raise RuntimeError(f"Echo cadence: latent {frame_index} arrived with pending "
                                   f"{self._echo_pending_ids} (expected {expected})")
            self._echo_pending_ids.append(frame_index)
            self._echo_pending_q.append(q.detach())
            self._echo_pending_k.append(k.detach())
            self._echo_pending_v.append(v.detach())
        if len(self._echo_pending_ids) == cad:
            self._echo_commit(torch.cat(self._echo_pending_q, dim=1),
                              torch.cat(self._echo_pending_k, dim=1),
                              torch.cat(self._echo_pending_v, dim=1), group_start, frame_seqlen)
            self._echo_pending_q, self._echo_pending_k, self._echo_pending_v = [], [], []
            self._echo_pending_ids = []

    def _echo_spectral_anchor(self, q, local, grid_sizes, freqs, frame_index, frame_seqlen):
        # Six fixed early generated frames; views reuse existing Echo storage.
        # Not cyclic anchors, not another persistent KV cache. Exclude source 0.
        ticks = getattr(self, 'echo_spectral_anchor_frames', None)
        if ticks is None:
            # Legacy: the first frame of each of the first six generated ENTRIES,
            # i.e. block-denominated (latents 1-6 framewise, 1,4,..,16 at chunk 3, and only
            # three entries -> RuntimeError at chunk 6).
            chosen = [i for i, tick in enumerate(self._echo_early_ids) if tick > 0][:6]
            if len(chosen) != 6:
                raise RuntimeError('spectral anchor requires six clean early blocks')
            ak = torch.cat([self._echo_early_k[i][:, :frame_seqlen] for i in chosen], dim=1)
            av = torch.cat([self._echo_early_v[i][:, :frame_seqlen] for i in chosen], dim=1)
        else:
            # Factorial: the same ABSOLUTE latent frames at every granularity, each
            # sliced out of whichever early Echo entry contains it.
            parts_k, parts_v = [], []
            for t in ticks:
                for tick, ek, ev in zip(self._echo_early_ids, self._echo_early_k,
                                        self._echo_early_v):
                    if tick <= t < tick + ek.shape[1] // frame_seqlen:
                        o = (t - tick) * frame_seqlen
                        parts_k.append(ek[:, o:o + frame_seqlen])
                        parts_v.append(ev[:, o:o + frame_seqlen])
                        break
                else:
                    raise RuntimeError(f'spectral anchor frame {t} is not in the Echo early pool')
            if len(parts_k) != 6:
                raise RuntimeError('spectral anchor requires six anchor frames')
            ak, av = torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)
        ag = grid_sizes.clone()
        ag[:, 0] = 6
        rq = block_relativistic_rope(q, grid_sizes, freqs, 6,
                                     theta_by_head=self.rope_theta_by_head).type_as(av)
        rk = block_relativistic_rope(ak, ag, freqs, 0,
                                     theta_by_head=self.rope_theta_by_head).type_as(av)
        anchor = attention(rq, rk, av)
        out = spectral_anchor_blend(local, anchor, grid_sizes, self.echo_spectral_strength)
        self._echo_spectral_calls += 1
        if self._echo_spectral_first_frame is None:
            self._echo_spectral_first_frame = int(frame_index)
        return out

    def _echo_cached_attention(self, q, k, v, grid_sizes, freqs,
                               current_start_frame, frame_seqlen):
        # factorial.echo_cadence_latents: rotation / commit keyed by the cadence-group start
        # (== current_start_frame when the cadence is off or for the source latent 0).
        cad = int(getattr(self, 'echo_cadence_latents', 0) or 0)
        group_start = current_start_frame
        if cad and current_start_frame >= 1:
            group_start = current_start_frame - (current_start_frame - 1) % cad
        self._echo_hw = (int(grid_sizes[0][1]), int(grid_sizes[0][2]))   # H2: token grid for correspondence
        pin = getattr(self, 'echo_pin_latent', None)
        if pin is None:
            anchor_ids = self._echo_anchor_indices(group_start)
        else:
            anchor_ids = self._echo_pin_anchor_indices(group_start, frame_seqlen)
        parts_k = [self._echo_early_k[i] for i in anchor_ids]
        parts_v = [self._echo_early_v[i] for i in anchor_ids]
        # C2 / DyMoS: index (in parts) of the source entry (tick 0) when the rotation holds it
        src_part = next((j for j, i in enumerate(anchor_ids) if self._echo_early_ids[i] == 0), None)
        # accounting: the latent id of every attended key frame, in part order
        # (None = a compressed / blended frame); Python ints only, never a tensor the output uses
        parts_ids = [[self._echo_early_ids[i] + j for j in range(self._echo_early_k[i].shape[1] // frame_seqlen)]
                     for i in anchor_ids]
        reader = getattr(self, 'material_reader', None)
        def origin(part, tick):
            return (torch.arange(part.shape[1], device=part.device) +
                    tick * frame_seqlen)[None].expand(part.shape[0], -1)
        parts_origin = ([origin(self._echo_early_k[i], self._echo_early_ids[i])
                         for i in anchor_ids] if reader is not None else [])
        anchor_frame_ids = {self._echo_early_ids[i] for i in anchor_ids}
        if pin is not None:
            # Pinned slot: one latent frame, first (oldest) part of the anchor block, present in
            # every evaluation once that latent is committed. A whole-entry pin (the 1-frame
            # source entry) also dedups the recent store, exactly like an anchored entry.
            pinned = self._echo_pin_kv(frame_seqlen)
            if pinned is not None:
                pk, pv, ptick, whole = pinned
                parts_k.insert(0, pk)
                parts_v.insert(0, pv)
                parts_ids.insert(0, [int(pin)])
                if whole and ptick == 0:
                    src_part = 0                          # pinsrc: the source is the pinned part
                elif src_part is not None:
                    src_part += 1
                if reader is not None:
                    parts_origin.insert(0, origin(pk, int(pin)))
                if whole:
                    anchor_frame_ids.add(ptick)
        if self._echo_compressed_k is not None:
            parts_k.append(self._echo_compressed_k)
            parts_v.append(self._echo_compressed_v)
            parts_ids.append([None] * (self._echo_compressed_k.shape[1] // frame_seqlen))
            if reader is not None:
                parts_origin.append(self._echo_compressed_origin)
        for recent_id, recent_k, recent_v in zip(
                self._echo_recent_ids, self._echo_recent_k, self._echo_recent_v):
            if recent_id not in anchor_frame_ids:
                parts_k.append(recent_k)
                parts_v.append(recent_v)
                parts_ids.append([recent_id + j for j in range(recent_k.shape[1] // frame_seqlen)])
                if reader is not None:
                    parts_origin.append(origin(recent_k, recent_id))
        for pend_id, pend_k, pend_v in zip(          # cadence group so far (empty otherwise)
                self._echo_pending_ids, self._echo_pending_k, self._echo_pending_v):
            parts_k.append(pend_k)
            parts_v.append(pend_v)
            parts_ids.append([pend_id + j for j in range(pend_k.shape[1] // frame_seqlen)])
            if reader is not None:
                parts_origin.append(origin(pend_k, pend_id))
        parts_k.append(k)
        parts_v.append(v)
        parts_ids.append([current_start_frame + j for j in range(k.shape[1] // frame_seqlen)])
        if reader is not None:
            parts_origin.append(origin(k, current_start_frame))
        spatial_ids = None
        if getattr(self, 'echo_preserve_spatial', False):
            parts_spatial = []
            for part in parts_k:
                if part is self._echo_compressed_k:
                    ids = self._echo_compressed_spatial
                    if ids is None or ids.shape != part.shape[:2]:
                        raise RuntimeError("Echo compressed K/V lost spatial metadata")
                else:
                    ids = torch.arange(part.shape[1], device=part.device).remainder(frame_seqlen)
                    ids = ids[None, :].expand(part.shape[0], -1)
                parts_spatial.append(ids)
            spatial_ids = torch.cat(parts_spatial, dim=1)[:, -self.max_attention_size:]
        if pin is not None and sum(p.shape[1] for p in parts_k) > self.max_attention_size:
            # The cap below drops the OLDEST keys first, i.e. the pinned slot: never allowed.
            raise RuntimeError(f"Echo pin: {sum(p.shape[1] for p in parts_k) // frame_seqlen} "
                               f"frames > window {self.max_attention_size // frame_seqlen}")
        raw_k = torch.cat(parts_k, dim=1)[:, -self.max_attention_size:]
        all_v = torch.cat(parts_v, dim=1)[:, -self.max_attention_size:]
        frames = raw_k.shape[1] // frame_seqlen
        full_grid = grid_sizes.clone()
        full_grid[0][0] = frames
        query_start = max(0, frames - (q.shape[1] // frame_seqlen))
        rq = block_relativistic_rope(
            q, grid_sizes, freqs, query_start,
            theta_by_head=self.rope_theta_by_head).type_as(v)
        rk = block_relativistic_rope(
            raw_k, full_grid, freqs, 0,
            theta_by_head=self.rope_theta_by_head,
            spatial_ids=spatial_ids).type_as(v)
        gamma = _motion_gamma_for_step()
        src_entries = getattr(self, 'source_entries', None)
        ref_gamma = self._dymos_ref_gamma(current_start_frame)
        src_off = None
        if (src_entries is not None or ref_gamma) and src_part is not None:
            total = sum(p.shape[1] for p in parts_k)
            src_off = sum(p.shape[1] for p in parts_k[:src_part]) - max(0, total - self.max_attention_size)
            if src_off < 0:
                src_off = None                            # the cap dropped the source part
        kept = None
        port = None
        if (getattr(self, 'forcing_kv', None) is not None or getattr(self, 'future_kv', None) is not None) \
                and current_start_frame > 0:
            # ports (Forcing-KV / Future Forcing) on Echo + pinsrc: read-time views of the Echo window (the source
            # is the pinned first part; compressed parts carry no latent id and are not attended)
            soft = None
            seq_pack = [i for p in parts_ids for i in p][-(self.max_attention_size // frame_seqlen):]
            if getattr(self, 'forcing_kv', None) is not None:
                out, rk, all_v, kept, n_dyn = self._fkv_attention(rq, rk, all_v, seq_pack, current_start_frame,
                                                                  frame_seqlen)
                port = dict(name="forcing_kv", dynamic_tokens=n_dyn)
            else:
                out, rk, all_v, kept = self._ffk_attention(q, rq, rk, all_v, seq_pack, current_start_frame, frame_seqlen,
                                                           grid_sizes, freqs, query_start)
                port = dict(name="future_kv", **self._ff_state.get("last", {}))
            src_off = 0
            if _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                self.pack_stats.append(int(rk.shape[1]))
        elif (getattr(self, 'packcache_decay', None) is not None or getattr(self, 'fp_pool', None) is not None
                or getattr(self, 'forget', None) is not None) \
                and current_start_frame > 0:
            # ports: the window is subsampled / pooled at read time (pinsrc keeps the
            # source part full). composition (factorial.compose): with a C2 soft entry or a DyMoS
            # bias, the source part of the PACKED window (always whole, condition anchor) is routed exactly like the
            # unpacked window: the soft entry is roped at the source part's ORIGINAL window position (packing keeps
            # every kept token's RoPE) and substituted at its packed offset; the DyMoS bias sits at that offset.
            soft = None
            if src_entries is not None and src_off is not None:
                soft_k, soft_v = src_entries["soft"]
                one = grid_sizes.clone()
                one[0][0] = 1
                soft = (block_relativistic_rope(soft_k, one, freqs, src_off // frame_seqlen,
                                                theta_by_head=self.rope_theta_by_head).type_as(v), soft_v)
            seq_pack = [i for p in parts_ids for i in p][-(self.max_attention_size // frame_seqlen):]
            rk, all_v, src_off, kept = self._pack_window(rk, all_v, seq_pack, current_start_frame,
                                                         frame_seqlen, grid_sizes)
            if _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                self.pack_stats.append(int(rk.shape[1]))
            if (src_entries is not None or ref_gamma) and src_off is not None:
                out = self._source_routed_attention(rq, rk, all_v, src_off, frame_seqlen, ref_gamma, soft)
            elif getattr(self, 'locality', None) is not None:
                out = _frame_bias_attention(rq, rk, all_v, self._locality_mask(        # locality prior
                    int(rq.shape[1]), frame_seqlen, grid_sizes, rq.device))
            elif getattr(self, '_pack_bias', None) is not None:
                out = _frame_bias_attention(rq, rk, all_v, self._pack_bias)   # fgs_soft
            else:
                out = attention(rq, rk, all_v)
            #: auxiliary static-query read; H1-FM remains the default kernel path.
            query_alpha = (getattr(self, "forget", None) or {}).get("prev_query_alpha")
            if query_alpha is not None:
                from .static_fidelity import static_query_read
                out = static_query_read(out, rq, rk, all_v, kept, current_start_frame, frame_seqlen,
                                        getattr(self, "forget_static_conf", None), query_alpha, attention)
        elif src_off is not None:
            soft = None
            if src_entries is not None:
                # the soft entry is stored raw like every Echo entry: rope it alone at the source
                # part's window position (RoPE is per token, so this equals roping it inside the
                # window; tests/test_factorial.py SourceSchedule checks the equality)
                soft_k, soft_v = src_entries["soft"]
                one = grid_sizes.clone()
                one[0][0] = 1
                soft = (block_relativistic_rope(soft_k, one, freqs, src_off // frame_seqlen,
                                                theta_by_head=self.rope_theta_by_head).type_as(v),
                        soft_v)
            out = self._source_routed_attention(rq, rk, all_v, src_off, frame_seqlen, ref_gamma, soft)
        else:
            soft = None
            out = (_dymos_attention(rq, rk, all_v, frame_seqlen, gamma)
                   if gamma != 0.0 and current_start_frame > 0
                   else attention(rq, rk, all_v))
        if getattr(self, 'context_log', None) is not None and current_start_frame > 0 \
                and _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
            seq = [i for p in parts_ids for i in p][-(self.max_attention_size // frame_seqlen):]
            if kept is None and src_off is None and 0 in seq:
                src_off = seq.index(0) * frame_seqlen
            self._context_record(seq, current_start_frame, q.shape[1] // frame_seqlen, rq, rk, src_off,
                                 frame_seqlen, soft, ref_gamma, kept=kept)
            if port is not None:
                self._port_record(port, kept, frame_seqlen)
        # First/high-noise denoising evaluation only; never on the clean refresh.
        if (getattr(self, 'echo_spectral_strength', 0.0) > 0 and
                current_start_frame > 21 and _DENOISE_STEP_FRACTION == 0.0):
            out = self._echo_spectral_anchor(q, out, grid_sizes, freqs,
                                             current_start_frame, frame_seqlen)
        if reader is not None and current_start_frame > 21 and _DENOISE_STEP_FRACTION is not None:
            origins = torch.cat(parts_origin, dim=1)[:, -self.max_attention_size:]
            out = reader(self, rq, rk, all_v, origins, freqs, grid_sizes,
                         current_start_frame, out)
        if _DENOISE_STEP_FRACTION is None:
            if cad and current_start_frame >= 1:
                self._echo_cadence_push(q, k, v, current_start_frame, group_start, frame_seqlen)
            else:
                self._echo_commit(q, k, v, current_start_frame, frame_seqlen)
        return out

    # --- B9: checkpoint-safe cache (opt-in, `checkpoint_safe_cache`) -----------
    # torch.utils.checkpoint re-runs a block's forward during backward. With a mutable
    # cache that re-run reads the cache / Echo memory as it is AFTER the whole rollout
    # (rolled at block 7, memory evolved) -> CheckpointError, and it writes into the
    # committed cache again. In this mode each cache-attention call that carries gradient
    #   * records the state it reads (KV tensors by reference + end indices; Echo memory as
    #     list copies -- its tensors are never written in place, only q_center is cloned),
    #   * runs the unchanged `_cache_attention` on a private copy of the KV tensors and then
    #     commits only the new tokens to the live cache (values, never autograd history),
    # and the backward re-run replays `_cache_attention` against the recorded state on a
    # throw-away copy, so it sees exactly the forward-time context and commits nothing.
    # Evicting calls (any grad mode) replace the live KV tensors instead of rolling them in
    # place (copy on write), which is what keeps the recorded references valid.
    _ECHO_STATE_ATTRS = (
        "_echo_early_k", "_echo_early_v", "_echo_early_ids",
        "_echo_recent_k", "_echo_recent_v", "_echo_recent_ids",
        "_echo_compressed_k", "_echo_compressed_v", "_echo_compressed_spatial",
        "_echo_compressed_origin", "_echo_q_center", "_echo_q_count",
        "_echo_spectral_calls", "_echo_spectral_first_frame",
        "_echo_pending_q", "_echo_pending_k", "_echo_pending_v", "_echo_pending_ids")

    def _echo_state_copy(self, state=None):
        """Copy of the Echo memory safe to mutate: lists copied, q_center (updated in place
        by _echo_commit) cloned; the stored K/V tensors are only ever replaced, so shared."""
        if state is None:
            state = {a: getattr(self, a) for a in self._ECHO_STATE_ATTRS}
        out = {}
        for a, val in state.items():
            if isinstance(val, list):
                val = list(val)
            elif a == "_echo_q_center" and val is not None:
                val = val.clone()
            out[a] = val
        return out

    def _cache_need_evict(self, kv_cache, num_new_tokens, current_end):
        """The eviction predicate of `_cache_attention` (plain / relativistic paths)."""
        return (self.local_attn_size != -1
                and current_end > kv_cache["global_end_index"].item()
                and num_new_tokens + kv_cache["local_end_index"].item() > kv_cache["k"].shape[1])

    def _safe_cache_attention(self, q, k, v, grid_sizes, freqs, kv_cache, current_start,
                              cache_record=None):
        global _DENOISE_STEP_FRACTION
        if cache_record is not None and "pre" in cache_record:
            # Checkpoint recomputation: replay against the recorded forward-time state.
            pre = cache_record["pre"]
            scratch = {
                "k": pre["k"].clone(), "v": pre["v"].clone(),
                "global_end_index": torch.tensor([pre["global_end"]], dtype=torch.long,
                                                 device=pre["k"].device),
                "local_end_index": torch.tensor([pre["local_end"]], dtype=torch.long,
                                                device=pre["k"].device)}
            live_echo = self._echo_state_copy() if self.echo_memory else None
            live_frac = _DENOISE_STEP_FRACTION
            if self.echo_memory:
                for a, val in self._echo_state_copy(pre["echo"]).items():
                    setattr(self, a, val)
            _DENOISE_STEP_FRACTION = pre["denoise_frac"]
            try:
                return self._cache_attention(q, k, v, grid_sizes, freqs, scratch, current_start)
            finally:
                _DENOISE_STEP_FRACTION = live_frac
                if self.echo_memory:
                    for a, val in live_echo.items():
                        setattr(self, a, val)

        if cache_record is not None:
            # Original forward under checkpoint: record the state this call reads. The KV
            # tensors are kept by reference: later calls only write positions >= this
            # call's new tokens in place, and evictions replace the tensors (below).
            kv_cache["checkpoint_records"] = True
            cache_record["pre"] = {
                "k": kv_cache["k"].detach(), "v": kv_cache["v"].detach(),
                "global_end": int(kv_cache["global_end_index"].item()),
                "local_end": int(kv_cache["local_end_index"].item()),
                "echo": self._echo_state_copy() if self.echo_memory else None,
                "denoise_frac": _DENOISE_STEP_FRACTION}

        if self.echo_memory:
            # Echo never reads the KV tensors (only fills the end indices) and commits
            # detached K/V, so the live call needs no private copy.
            return self._cache_attention(q, k, v, grid_sizes, freqs, kv_cache, current_start)

        num_new_tokens = q.shape[1]
        evict = self._cache_need_evict(kv_cache, num_new_tokens, current_start + num_new_tokens)
        grad = torch.is_grad_enabled()
        if not grad and not (evict and kv_cache.get("checkpoint_records", False)):
            return self._cache_attention(q, k, v, grid_sizes, freqs, kv_cache, current_start)

        scratch = {"k": kv_cache["k"].detach().clone(), "v": kv_cache["v"].detach().clone(),
                   "global_end_index": kv_cache["global_end_index"].clone(),
                   "local_end_index": kv_cache["local_end_index"].clone()}
        x = self._cache_attention(q, k, v, grid_sizes, freqs, scratch, current_start)
        with torch.no_grad():
            if evict:
                # copy on write; under grad the scratch is held by autograd, so copy again
                kv_cache["k"] = scratch["k"].detach().clone() if grad else scratch["k"]
                kv_cache["v"] = scratch["v"].detach().clone() if grad else scratch["v"]
            else:
                end = int(scratch["local_end_index"].item())
                start = end - num_new_tokens
                kv_cache["k"][:, start:end] = scratch["k"][:, start:end].detach()
                kv_cache["v"][:, start:end] = scratch["v"][:, start:end].detach()
            kv_cache["global_end_index"].copy_(scratch["global_end_index"])
            kv_cache["local_end_index"].copy_(scratch["local_end_index"])
        return x

    def _cache_attention(self, q, k, v, grid_sizes, freqs, kv_cache, current_start):
        """KV-cache / Echo self-attention (the body of forward's cache branch, unchanged).
        Reads AND mutates the cache (and Echo memory) in place; returns the attention
        output [B, L, n, d] before the output projection."""
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        num_new_tokens = q.shape[1]
        current_end = current_start + num_new_tokens
        if self.echo_memory:
            x = self._echo_cached_attention(
                q, k, v, grid_sizes, freqs, current_start_frame, frame_seqlen)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(min(current_end, kv_cache["k"].shape[1]))
            return x
        sink_tokens = self.sink_size * frame_seqlen
        # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        kv_cache_size = kv_cache["k"].shape[1]
        need_evict = self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)

        if self.relativistic_rope:
            # ---- Steady-forcing path -------------------------------------
            # Cache RAW keys/values and re-rope the whole window relatively
            # every step. Combined with the dual-sink + EMA cache update this
            # keeps a static fixed-camera scene steady over long horizons.
            num_new_frames = num_new_tokens // frame_seqlen
            max_attention_frames = self.max_attention_size // frame_seqlen
            if need_evict:
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                # Dual-sink + EMA-sink: blend evicted tokens into the soft-sink
                # zone. Frame 0 stays a hard anchor; the EMA spans the remaining
                # sink frames [frame_seqlen, sink_tokens).
                if self.ema_sink and sink_tokens > 0:
                    evicted_k = kv_cache["k"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    evicted_v = kv_cache["v"][:, sink_tokens:sink_tokens + num_evicted_tokens].clone()
                    n = min(num_evicted_tokens, sink_tokens - frame_seqlen)
                    if n > 0:
                        a = self.compression_alpha
                        kv_cache["k"][:, frame_seqlen:frame_seqlen + n] = \
                            a * kv_cache["k"][:, frame_seqlen:frame_seqlen + n] + (1 - a) * evicted_k[:, :n]
                        kv_cache["v"][:, frame_seqlen:frame_seqlen + n] = \
                            a * kv_cache["v"][:, frame_seqlen:frame_seqlen + n] + (1 - a) * evicted_v[:, :n]

                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k  # raw K
                kv_cache["v"][:, local_start_index:local_end_index] = v

                k_for_rope = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                grid_sizes_full = grid_sizes.clone()
                grid_sizes_full[0][0] = max_attention_frames
                relative_start_frame = max_attention_frames - num_new_frames
                if getattr(self, 'guidance_active', False) and not _BRANCH_READONLY:
                    self._guidance_base_qstart = relative_start_frame      # (see the else branch)
                roped_query = block_relativistic_rope(
                    q, grid_sizes, freqs, start_frame=relative_start_frame,
                    theta_by_head=self.rope_theta_by_head).type_as(v)
                roped_key = block_relativistic_rope(
                    k_for_rope, grid_sizes_full, freqs, start_frame=0,
                    theta_by_head=self.rope_theta_by_head).type_as(v)
                # frame 0 hard anchor: keep its stored (roped-at-0) key untouched.
                # pure_sliding_window (opt-in, sink_size 0): slot 0 holds the RAW key of
                # the oldest window frame once frame 0 is evicted, so it is re-roped like
                # every other window frame instead of entering attention unroped.
                #: with a per-layer read window shorter than the buffer
                # (factorial.local_layer_sizes) the window no longer starts at slot 0, so there
                # is no stored anchor to restore and every (raw) key is re-roped. On every other
                # path the read window equals the buffer, the start is 0 and this is unchanged.
                if not getattr(self, 'pure_sliding_window', False) and \
                        local_end_index <= self.max_attention_size:
                    roped_key[:, :frame_seqlen] = k_for_rope[:, :frame_seqlen]
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k  # raw K
                kv_cache["v"][:, local_start_index:local_end_index] = v

                k_for_rope = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                grid_sizes_full = grid_sizes.clone()
                grid_sizes_full[0][0] = min(local_end_index // frame_seqlen, max_attention_frames)
                relative_start_frame = current_start_frame if current_start_frame < max_attention_frames \
                    else max_attention_frames - num_new_frames
                if getattr(self, 'guidance_active', False):
                    #: the read-only prompt-extrapolation branch re-runs an evaluation whose base pass
                    # may have taken the EVICT branch (query at window - k); this branch would put the query at the
                    # absolute index while it is < the window (pre-existing behaviour of the default path's later
                    # evaluations of the first evicting block, left unchanged). The alternative pass reuses the base
                    # pass's query position so both branches see exactly the same geometry.
                    if _BRANCH_READONLY:
                        relative_start_frame = self._guidance_base_qstart
                    else:
                        self._guidance_base_qstart = relative_start_frame
                roped_query = block_relativistic_rope(
                    q, grid_sizes, freqs, start_frame=relative_start_frame,
                    theta_by_head=self.rope_theta_by_head).type_as(v)
                roped_key = block_relativistic_rope(
                    k_for_rope, grid_sizes_full, freqs, start_frame=0,
                    theta_by_head=self.rope_theta_by_head).type_as(v)
                if getattr(self, 'pure_sliding_window', False):
                    pass  # raw K in every slot; the whole window is re-roped by position
                elif local_start_index == 0:
                    # store the roped frame-0 anchor so later steps read it back
                    kv_cache["k"][:, :frame_seqlen] = roped_key[:, :frame_seqlen]
                elif local_end_index <= self.max_attention_size:   # window starts at slot 0 (§17.10)
                    roped_key[:, :frame_seqlen] = k_for_rope[:, :frame_seqlen]

            _capture_anchor_attention(roped_query, roped_key, frame_seqlen,
                                      sink_tokens, current_start_frame)
            _v_win = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            motion_gamma = _motion_gamma_for_step()
            src_entries = getattr(self, 'source_entries', None)
            ref_gamma = self._dymos_ref_gamma(current_start_frame)
            ctx_k, ctx_v, ctx_src, kept = roped_key, _v_win, None, None
            port = None
            tc_out = None
            if getattr(self, 'tether_cache', None) is not None and current_start_frame > 0:
                # TetherCache port: None during warm-up (the host path below runs unchanged)
                tc_out = self._tc_attention(q, k_for_rope, _v_win,
                                            self._sink_window_ids(current_start_frame, num_new_frames,
                                                                  roped_key.shape[1] // frame_seqlen),
                                            current_start_frame, frame_seqlen, grid_sizes, freqs)
            if tc_out is not None:
                x, roped_query, ctx_k, kept = tc_out
                ctx_src = 0
                port = dict(name="tether_cache", memory=list(self._tc_state["mem_ids"]))
                if _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                    self.pack_stats.append(int(ctx_k.shape[1]))
            elif (getattr(self, 'forcing_kv', None) is not None or getattr(self, 'future_kv', None) is not None) \
                    and current_start_frame > 0:
                # ports (Forcing-KV / Future Forcing): read-time views of the sink window (slot 0 = the source)
                seq_pack = self._sink_window_ids(current_start_frame, num_new_frames, roped_key.shape[1] // frame_seqlen)
                if getattr(self, 'forcing_kv', None) is not None:
                    x, ctx_k, ctx_v, kept, n_dyn = self._fkv_attention(roped_query, roped_key, _v_win, seq_pack,
                                                                       current_start_frame, frame_seqlen)
                    port = dict(name="forcing_kv", dynamic_tokens=n_dyn)
                else:
                    x, ctx_k, ctx_v, kept = self._ffk_attention(q, roped_query, roped_key, _v_win, seq_pack,
                                                                current_start_frame, frame_seqlen, grid_sizes, freqs,
                                                                relative_start_frame)
                    port = dict(name="future_kv", **self._ff_state.get("last", {}))
                ctx_src = 0
                if _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                    self.pack_stats.append(int(ctx_k.shape[1]))
            elif (getattr(self, 'packcache_decay', None) is not None or getattr(self, 'fp_pool', None) is not None
                or getattr(self, 'forget', None) is not None) \
                    and current_start_frame > 0:
                # ports (read-time subsampling / pooling of the window; slot 0 = source full)
                seq_pack = self._sink_window_ids(current_start_frame, num_new_frames,
                                                 roped_key.shape[1] // frame_seqlen)
                #: on the dual sink the EMA soft-sink slots (no latent id) are kept full
                # unless factorial.pack_soft_sink=budget
                keep_full = {j for j, fid in enumerate(seq_pack) if fid is None} \
                    if getattr(self, 'pack_soft_sink', None) == 'full' else ()
                ctx_k, ctx_v, ctx_src, kept = self._pack_window(roped_key, _v_win, seq_pack, current_start_frame,
                                                                frame_seqlen, grid_sizes, keep_full=keep_full)
                if _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                    self.pack_stats.append(int(ctx_k.shape[1]))
                if (src_entries is not None or ref_gamma) and ctx_src is not None:
                    # composition: slot 0 (the source, roped at frame 0, never thinned) stays at packed
                    # offset ctx_src; the stored-form soft entry / the DyMoS bias act there, as in the unpacked window
                    x = self._source_routed_attention(
                        roped_query, ctx_k, ctx_v, ctx_src, frame_seqlen, ref_gamma,
                        None if src_entries is None else src_entries["soft"])
                elif getattr(self, 'locality', None) is not None:
                    x = _frame_bias_attention(roped_query, ctx_k, ctx_v, self._locality_mask(   # locality prior
                        int(roped_query.shape[1]), frame_seqlen, grid_sizes, roped_query.device))
                elif getattr(self, '_pack_bias', None) is not None:
                    x = _frame_bias_attention(roped_query, ctx_k, ctx_v, self._pack_bias)   # fgs_soft
                else:
                    x = attention(roped_query, ctx_k, ctx_v)
            elif (src_entries is not None or ref_gamma) and current_start_frame > 0:
                # sink policies (sink_size >= 1): the source is the hard anchor in window slot 0
                # (stored roped at frame 0); _v_win is a VIEW of the cache, so the soft entry is
                # only ever placed into the fresh window copy built inside the routed call.
                x = self._source_routed_attention(
                    roped_query, roped_key, _v_win, 0, frame_seqlen, ref_gamma,
                    None if src_entries is None else src_entries["soft"])
            elif motion_gamma != 0.0 and current_start_frame > 0:
                x = _dymos_attention(roped_query, roped_key, _v_win, frame_seqlen, motion_gamma)
            else:
                x = attention(roped_query, roped_key, _v_win)
            if getattr(self, 'context_log', None) is not None and current_start_frame > 0 \
                    and _DENOISE_STEP_FRACTION == 0.0 and not _BRANCH_READONLY:
                # attended frame identity of the sink window: _sink_window_ids
                seq = self._sink_window_ids(current_start_frame, num_new_frames, roped_key.shape[1] // frame_seqlen,
                                            n_buf=local_end_index // frame_seqlen)
                if tc_out is not None:
                    seq = [f for f, _ in kept]                # TetherCache: the port's own window
                if kept is None:
                    ctx_src = 0 if (int(self.sink_size) >= 1 and not getattr(self, 'pure_sliding_window', False)
                                    and local_end_index <= self.max_attention_size) else None
                self._context_record(seq, current_start_frame, num_new_frames, roped_query, ctx_k, ctx_src,
                                     frame_seqlen, None if src_entries is None else src_entries["soft"], ref_gamma,
                                     kept=kept)
                if port is not None:
                    self._port_record(port, kept, frame_seqlen)
        else:
            # ---- Original cache path (rope-then-cache, single sink) -------
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            if need_evict:
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            _k_win = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            _v_win = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            _capture_anchor_attention(roped_query, _k_win, frame_seqlen,
                                      sink_tokens, current_start_frame)
            motion_gamma = _motion_gamma_for_step()
            if motion_gamma != 0.0 and current_start_frame > 0:
                x = _dymos_attention(roped_query, _k_win, _v_win, frame_seqlen, motion_gamma)
            else:
                x = attention(roped_query, _k_win, _v_win)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return x

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        cache_record=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                # `[:-0]` is an EMPTY slice: when the token count is already a multiple of 128
                # (e.g. 16 latent frames under teacher forcing) the unguarded slice dropped
                # everything. Match the guarded non-TF branch.
                if padded_length > 0:
                    x = x[:, :, :-padded_length]
                x = x.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                _attn = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                if padded_length > 0:
                    _attn = _attn[:, :, :-padded_length]
                x = _attn.transpose(2, 1)
        else:
            if getattr(self, "checkpoint_safe_cache", False):
                x = self._safe_cache_attention(q, k, v, grid_sizes, freqs, kv_cache,
                                               current_start, cache_record)
            else:
                x = self._cache_attention(q, k, v, grid_sizes, freqs, kv_cache, current_start)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 frame_seq_length=None,
                 relativistic_rope=False,
                 ema_sink=False,
                 compression_alpha=0.99,
                 rope_jitter_sigma=0.0,
                 rope_jitter_seed=0,
                 echo_memory=False,
                 echo_anchor_pool=18,
                 echo_anchor_frames=12,
                 echo_compressed_frames=3,
                 echo_recent_frames=6,
                 echo_anchor_stride=3,
                 echo_preserve_spatial=False,
                 echo_spectral_strength=0.0):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps,
            relativistic_rope=relativistic_rope, ema_sink=ema_sink,
            compression_alpha=compression_alpha, frame_seq_length=frame_seq_length,
            rope_jitter_sigma=rope_jitter_sigma, rope_jitter_seed=rope_jitter_seed,
            echo_memory=echo_memory, echo_anchor_pool=echo_anchor_pool,
            echo_anchor_frames=echo_anchor_frames,
            echo_compressed_frames=echo_compressed_frames,
            echo_recent_frames=echo_recent_frames,
            echo_anchor_stride=echo_anchor_stride,
            echo_preserve_spatial=echo_preserve_spatial,
            echo_spectral_strength=echo_spectral_strength)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        cache_record=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            cache_record=cache_record)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 relativistic_rope=False,
                 ema_sink=False,
                 compression_alpha=0.99,
                 frame_seq_length=None,
                 rope_jitter_sigma=0.0,
                 rope_jitter_seed=0,
                 echo_memory=False,
                 echo_anchor_pool=18,
                 echo_anchor_frames=12,
                 echo_compressed_frames=3,
                 echo_recent_frames=6,
                 echo_anchor_stride=3,
                 echo_preserve_spatial=False,
                 echo_spectral_strength=0.0):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            relativistic_rope (`bool`, *optional*, defaults to False):
                Steady-forcing: cache raw keys and re-rope the attention window
                on a relative temporal frame each step. When False the original
                rope-then-cache path is used.
            ema_sink (`bool`, *optional*, defaults to False):
                Steady-forcing: on KV-cache rollover, EMA-blend evicted tokens
                into the soft-sink zone (dual-sink). Only effective together with
                `relativistic_rope`.
            compression_alpha (`float`, *optional*, defaults to 0.99):
                EMA retention rate used by `ema_sink` (higher = slower drift).
        """

        super().__init__()

        # Wan2.2 adds 'ti2v'. Structurally it is t2v INSIDE the network: Wan2.2's own
        # model.py gives img_emb / i2v_cross_attn only to model_type == 'i2v'
        # (wan/modules/model.py:437), and TI2V conditions externally by masking the image
        # latent into the first frame (wan/textimage2video.py:551,
        # `latent = (1-mask)*z + mask*latent`). So accept it and route it down the t2v
        # path -- no architectural change, and crucially no CLIP image encoder.
        # tokens per latent frame: 1560 for Wan2.1 @480x832, 390 for Wan2.2 (stride
        # (4,16,16) -> 30x52 latents -> 15*26). Drives the attention window and the
        # KV-cache eviction arithmetic, so it must follow the VAE, not be assumed.
        if frame_seq_length is None:
            frame_seq_length = int(os.environ.get("WAN_FRAME_SEQ_LENGTH", 1560))
        self.frame_seq_length = frame_seq_length
        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type
        self.is_ti2v = (model_type == 'ti2v')

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.relativistic_rope = relativistic_rope
        self.ema_sink = ema_sink
        self.compression_alpha = compression_alpha

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 't2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,
                                    relativistic_rope=relativistic_rope, ema_sink=ema_sink,
                                    compression_alpha=compression_alpha,
                                    frame_seq_length=frame_seq_length,
                                    rope_jitter_sigma=rope_jitter_sigma,
                                    rope_jitter_seed=rope_jitter_seed + layer_index,
                                    echo_memory=echo_memory,
                                    echo_anchor_pool=echo_anchor_pool,
                                    echo_anchor_frames=echo_anchor_frames,
                                    echo_compressed_frames=echo_compressed_frames,
                                    echo_recent_frames=echo_recent_frames,
                                    echo_anchor_stride=echo_anchor_stride,
                                    echo_preserve_spatial=echo_preserve_spatial,
                                    echo_spectral_strength=echo_spectral_strength)
            for layer_index in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
        # B9 (opt-in): gradient checkpointing through the mutable KV cache /
        # Echo memory. See CausalWanSelfAttention._safe_cache_attention.
        self.checkpoint_safe_cache = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def set_checkpoint_safe_cache(self, value: bool = True):
        """ B9: make cache-path training exact under gradient checkpointing.

        On: every gradient-carrying cache call records the cache / Echo state it reads and
        commits only values (no autograd history) to the live cache; the checkpoint
        recomputation replays against the recorded state without touching the live cache;
        evictions copy on write; and grad-enabled calls without checkpointing recompute the
        cross-attention K/V with gradient (as the checkpointed path always did) instead of
        reading the detached cross-attention cache. Off (default): unchanged behaviour.
        """
        self.checkpoint_safe_cache = bool(value)
        for block in self.blocks:
            block.self_attn.checkpoint_safe_cache = bool(value)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, first_block_size=None
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block intervals in TOKENS over one copy of the sequence. With first_block_size
        # the first block is that many frames (1 for i2v) and the rest are
        # num_frame_per_block; otherwise every block is num_frame_per_block.
        blocks = []
        f = 0
        if first_block_size:
            blocks.append((0, first_block_size * frame_seqlen)); f = first_block_size
        while f < num_frames:
            n = min(num_frame_per_block, num_frames - f)
            blocks.append((f * frame_seqlen, (f + n) * frame_seqlen)); f += n

        # clean context frames: block-causal within the clean half
        for (start, end) in blocks:
            context_ends[start:end] = end

        # noisy frames: attend to their own noisy block + all clean blocks BEFORE theirs
        for (cs, ce) in blocks:
            start, end = clean_ends + cs, clean_ends + ce
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            noise_context_ends[start:end] = cs

        def make_mask_mod(ce, ncs, nce, nns, nne):
            def attention_mask(b, h, q_idx, kv_idx):
                # clean frames: block-causal within the clean half
                clean_mask = (q_idx < clean_ends) & (kv_idx < ce[q_idx])
                # noisy frames attend to all preceding clean blocks + their own noisy block
                C1 = (kv_idx < nne[q_idx]) & (kv_idx >= nns[q_idx])
                C2 = (kv_idx < nce[q_idx]) & (kv_idx >= ncs[q_idx])
                noise_mask = (q_idx >= clean_ends) & (C1 | C2)
                eye_mask = q_idx == kv_idx
                return eye_mask | clean_mask | noise_mask
            return attention_mask

        # Built on the CPU: _compile=False materialises the dense Q x KV grid, which for the
        # doubled teacher-forcing sequence (2 x 19 x 1560 tokens) is a ~3.3 GiB transient. On a
        # GPU shared with other jobs that spike failed as OOM / "CUDA driver error: invalid
        # argument". The compact BlockMask is then moved to the device; it is built once and
        # cached, so the CPU cost is paid once per process.
        dev_idx = (context_ends, noise_context_starts, noise_context_ends, noise_noise_starts, noise_noise_ends)
        cpu_idx = tuple(t.cpu() for t in dev_idx)
        block_mask = create_block_mask(make_mask_mod(*cpu_idx), B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device="cpu")
        block_mask = block_mask.to(device)
        # flex_attention is torch.compile'd and traces mask_mod: it must close over DEVICE
        # tensors, not the CPU copies used to build the block layout.
        block_mask.mask_mod = make_mask_mod(*[t.to(device) for t in dev_idx])
        attention_mask = block_mask.mask_mod

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                                padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        safe = getattr(self, "checkpoint_safe_cache", False)
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                if safe:
                    # one record per (call, layer): filled by the forward, read by the
                    # recomputation B9)
                    kwargs["cache_record"] = {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": (
                            None if safe and torch.is_grad_enabled()
                            else crossattn_cache[block_index]),
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                kwargs.pop("cache_record", None)
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None: # TF
                # independent_first_frame (the i2v deployment) is a first block of ONE frame
                # followed by blocks of num_frame_per_block -- the same geometry
                # _prepare_blockwise_causal_attn_mask_i2v uses for inference. The TF mask
                # takes it as first_block_size so Resampling-Forcing training sees exactly
                # the block structure the student is deployed with.
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device, num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    first_block_size=1 if self.independent_first_frame else None
                )
            else: # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            # [1,32760,1536]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
