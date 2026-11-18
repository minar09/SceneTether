from typing import List, Optional
import time
import math
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
import tqdm

#: pre-registered half-width (log10-variance units) of the optional C3 uncertain band
# (`factorial.role_routing.uncertain: band`, launcher arm c123u): tokens within +-band of the threshold
# belong to neither variation class and follow the plain C2 schedule.
ROLE_UNCERTAIN_BAND_LOG10 = 0.15

#: Wan's official negative prompt as the native diagnostic uses it (Wan2.2 wan/configs/shared_config.py
# `sample_neg_prompt`, the text tools/baselines/dymos_native/negative_controls.py operates on). It carries the three
# static clauses 静态 / 静止 / 静止不动的画面; the repo's older configs hold Wan2.1's variant without 静态.
WAN22_SAMPLE_NEG_PROMPT = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，'
                           '低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，'
                           '形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')


def _negative_controls():
    """tools/baselines/dymos_native/negative_controls.py, reused unchanged and loaded by path."""
    import importlib.util
    import os as _os
    path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "tools", "baselines", "dymos_native", "negative_controls.py")
    spec = importlib.util.spec_from_file_location("negative_controls", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        # model_name selects the Wan family for the DiT, T5/tokenizer and VAE together.
        # They must agree: a Wan2.2 DiT with a Wan2.1 VAE would silently mismatch latent
        # channels (48 vs 16). Defaults to Wan2.1 when the config does not say.
        _mn = getattr(args, "model_name", None)
        _mk = dict(getattr(args, "model_kwargs", {}) or {})
        if _mn is not None:
            _mk.setdefault("model_name", _mn)
        self.generator = WanDiffusionWrapper(
            **_mk, is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder(_mn) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(_mn) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, the
        # first chunk uses the same schedule as the rest (backwards compatible).
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = torch.tensor(
                args.denoising_step_list_first_chunk, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list_first_chunk = timesteps[1000 - self.denoising_step_list_first_chunk]
        else:
            self.denoising_step_list_first_chunk = None

        # Both Wan2.1-1.3B and Wan2.2-TI2V-5B happen to have 30 blocks, but read it
        # rather than assume. frame_seq_length is NOT shared: 1560 for Wan2.1 @480x832,
        # 390 for Wan2.2 (stride (4,16,16)). It drives KV-cache sizing and eviction, so a
        # stale value addresses the wrong frames instead of erroring.
        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = int(getattr(
            self.generator.model.blocks[0].self_attn, "frame_seq_length", 1560))

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Timing state; populated only when `report_timing=True` is passed to
        # `inference()`. Kept as attributes so callers can read them afterward.
        self.last_generation_time = None
        self.first_chunk_time = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        # Report the *effective* attention config so a missing/unapplied
        # `model_kwargs` block is obvious (e.g. local_attn_size=-1 means no KV
        # eviction -> the fixed cache overflows past 21 frames for long videos).
        _sa = self.generator.model.blocks[0].self_attn
        print(
            f"[attn cfg] local_attn_size={_sa.local_attn_size} sink_size={_sa.sink_size} "
            f"relativistic_rope={_sa.relativistic_rope} ema_sink={_sa.ema_sink} "
            f"compression_alpha={_sa.compression_alpha}"
        )

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # Optional periodic KV-cache flush ("semantic purification"): every
        # `kv_flush_period` blocks, drop the accumulated middle context and keep
        # only the frame-0 anchor + the last `kv_flush_keep_frames` frames. This
        # pulls generation back toward the initial-quality regime to shed
        # accumulated artifacts/drift. Default OFF.
        #
        # SAFETY: only valid with relativistic_rope=True. That path caches *raw*
        # keys and re-ropes the window relatively at read time, so physically
        # moving keys to new slots is consistent. With the original path (roped
        # keys cached) moving keys would corrupt positions -> color flashing, so
        # we hard-disable the flush there.
        self.kv_flush_period = int(getattr(args, "kv_flush_period", 0))
        self.kv_flush_keep_frames = int(
            getattr(args, "kv_flush_keep_frames", self.num_frame_per_block))
        self._relativistic_rope = bool(
            getattr(self.generator.model.blocks[0].self_attn, "relativistic_rope", False))
        if self.kv_flush_period > 0 and not self._relativistic_rope:
            print("[kv_flush] disabled: requires model_kwargs.relativistic_rope: true "
                  "(raw-K cache). Flushing roped keys would cause color flashing.")
            self.kv_flush_period = 0
        if self.kv_flush_period > 0:
            print(f"[kv_flush] enabled: every {self.kv_flush_period} blocks, keep "
                  f"anchor + last {self.kv_flush_keep_frames} frames")

        # Phase-1 training-free mechanisms. All defaults are strict no-ops.
        p1 = getattr(args, "phase1", {}) or {}
        alg = p1.get("alg", {}) or {}
        dymos = p1.get("dymos", {}) or {}
        self.alg_enabled = bool(alg.get("enabled", False))
        self.alg_factor = float(alg.get("downsample_factor", 2.5))
        self.alg_step_ratio = float(alg.get("step_ratio", 0.1))
        self.alg_clean_delay_steps = int(alg.get("clean_delay_steps", 0))
        self._alg_clean_bank = self._alg_lowpass_bank = None
        if self.alg_enabled and self.alg_factor <= 1.0:
            raise ValueError("phase1.alg.downsample_factor must be > 1")
        if not 0.0 <= self.alg_step_ratio <= 1.0:
            raise ValueError("phase1.alg.step_ratio must be in [0,1]")
        from wan21.modules import causal_model as causal_attention
        causal_attention.set_motion_gamma(
            float(dymos.get("gamma", 0.0)),
            step_ratio=float(dymos.get("step_ratio", 0.1)))
        self._phase1_attention = causal_attention
        self.echo_enabled = bool(getattr(
            self.generator.model.blocks[0].self_attn, "echo_memory", False))
        lol_enabled = getattr(
            self.generator.model.blocks[0].self_attn, "rope_theta_by_head", None) is not None
        if self.echo_enabled and not self._relativistic_rope:
            raise ValueError("Echo-Forcing requires model_kwargs.relativistic_rope: true")
        if self.echo_enabled:
            # Echo entries are one k/v block each, so anchor/recent counts are in BLOCKS
            # while compressed is in frames. If the total exceeds the local window the
            # `[:, -max_attention_size:]` cap drops the oldest keys -- the anchors -- and
            # Echo silently degenerates to a sliding window. Shout, don't guess.
            sa = self.generator.model.blocks[0].self_attn
            nfpb = int(getattr(self, "num_frame_per_block", 1) or 1)
            budget = (sa.echo_anchor_frames + sa.echo_recent_past) * nfpb + \
                     sa.echo_compressed_frames + nfpb
            window = sa.max_attention_size // sa.frame_seq_length
            if budget > window:
                print(f"[phase1][WARNING] Echo frame budget {budget} > local window {window} "
                      f"at num_frame_per_block={nfpb}: anchors will be truncated away. "
                      f"Use a chunk-scaled overlay (configs/phase1/echo_forcing_chunk*.yaml).")
            else:
                print(f"[phase1] Echo frame budget {budget}/{window} at num_frame_per_block={nfpb}")
        if lol_enabled and not self._relativistic_rope:
            raise ValueError("LoL RoPE jitter requires model_kwargs.relativistic_rope: true")
        if self.alg_enabled or float(dymos.get("gamma", 0.0)) or lol_enabled or self.echo_enabled:
            print("[phase1] ALG={} DyMoS(gamma={}, lambda={}) LoL={} Echo={}".format(
                self.alg_enabled, float(dymos.get("gamma", 0.0)),
                float(dymos.get("step_ratio", 0.1)), lol_enabled, self.echo_enabled))
        self._init_factorial(args)

    def _init_factorial(self, args):
        """Granularity x memory factorial switches, tools/factorial/). Config block
        `factorial:`; every key absent/null/false is a strict no-op (tests/test_factorial.py).

        pure_sliding_window     true: sink_size 0 sliding window whose oldest frame is re-roped
                                like the others (the stock sink-0 path leaves slot 0 unroped).
        spectral_anchor_frames  list of 6 absolute latent ids for the spectral (FreqForcing-
                                style) anchor, so every granularity reads the same frames.
        warm_start_frames       K: the first K generated latents get
                                denoising_step_list_first_chunk, independent of block size; a
                                block straddling K is split at K (later boundaries unchanged).
        latent_indexed_renoise  true: the re-noise draw for latent f towards timestep t comes
                                from its own generator seeded by (renoise_seed, f, t), so it is
                                the same field at every granularity. inference.py sets
                                renoise_seed = seed + prompt index.
        warm_start_frames_nosplit  K: the first K GENERATED latents get
                                denoising_step_list_first_chunk, whole blocks only (a block is
                                warm iff it starts before K; K must be a multiple of the block
                                size, so the partition is untouched: K=6 -> blocks 1..6 at
                                nfpb 1, blocks 1-2 at 3, block 1 at 6 == the default). Exclusive
                                with warm_start_frames.
        echo_cadence_latents    c, Echo at nfpb 1 ONLY; refused at nfpb > 1 because
                                there it would change within-block causality,:
                                Echo commit / compression / anchor rotation fire once per c
                                generated latents ([1..c], [c+1..2c], ...), each commit being
                                the group's clean K/V as ONE entry keyed by the group start --
                                the entry shape of a chunk-c deployment, so use the chunk-c Echo
                                overlay (entry-denominated counts). Denoising stays per latent;
                                the group's earlier clean latents remain visible to later ones.
        echo_pin_source         true, Echo only, opt-in): the source latent 0 holds
                                ONE of the echo_anchor_frames anchor slots for the whole rollout
                                (first anchor part, one latent frame); the other A - 1 slots rotate
                                with the unchanged formula over the generated pool entries. Window
                                cap, compression, recent store, RoPE (order-based), sampler and
                                commits unchanged; the max attended frame count never exceeds the
                                window (21; checked here and asserted per call).
        echo_pin_latent         p >= 1: the matched replacement control -- the same slot holds
                                generated latent p (sliced out of its entry); the entry that IS
                                latent p (1-frame entry, nfpb 1) leaves the rotation, the source
                                stays in it. Exclusive with echo_pin_source.
        ema_alpha_normalize     {alpha_ref, k_ref},: EMA-sink
                                time-constant normalisation. model_kwargs.compression_alpha is
                                the RETENTION a of the soft-sink state per evicting block
                                (sink = a*sink + (1-a)*evicted; causal_model._cache_attention);
                                the new-state mixing weight is alpha = 1 - a.'s
                                alpha_k = 1 - (1 - alpha_ref)^(k/k_ref) on the mixing weight is
                                therefore a_k = a_ref^(k/k_ref) on compression_alpha. `alpha_ref`
                                is given in compression_alpha units (retention; must equal the
                                model's compression_alpha, default 0.99) so that k == k_ref is
                                bitwise the default; k_ref default 3. Information timing (which
                                frames are blended, when) is unchanged.
        """
        fac = getattr(args, "factorial", None) or {}
        sa = self.generator.model.blocks[0].self_attn
        self.pure_sliding_window = bool(fac.get("pure_sliding_window", False))
        if self.pure_sliding_window:
            if not self._relativistic_rope or self.echo_enabled or int(sa.sink_size) != 0:
                raise ValueError("factorial.pure_sliding_window needs relativistic_rope, no Echo "
                                 "and model_kwargs.sink_size: 0")
        spectral_frames = fac.get("spectral_anchor_frames", None)
        if spectral_frames is not None:
            spectral_frames = [int(t) for t in spectral_frames]
            if len(spectral_frames) != 6 or min(spectral_frames) < 1:
                raise ValueError("factorial.spectral_anchor_frames: six generated latent ids >= 1")
            if not float(getattr(sa, "echo_spectral_strength", 0.0)) > 0:
                raise ValueError("factorial.spectral_anchor_frames needs echo_spectral_strength > 0")
            if max(spectral_frames) >= int(sa.echo_anchor_pool):
                raise ValueError("factorial.spectral_anchor_frames must lie in the Echo early pool")
        if self.pure_sliding_window or spectral_frames is not None:
            for block in self.generator.model.blocks:
                attn = block.self_attn
                if self.pure_sliding_window:
                    attn.pure_sliding_window = True
                if spectral_frames is not None:
                    attn.echo_spectral_anchor_frames = list(spectral_frames)
        k = fac.get("warm_start_frames", None)
        self.warm_start_frames = None if k is None else int(k)
        if self.warm_start_frames is not None:
            if self.warm_start_frames < 1 or self.denoising_step_list_first_chunk is None:
                raise ValueError("factorial.warm_start_frames needs K >= 1 and "
                                 "denoising_step_list_first_chunk")
        self.latent_indexed_renoise = bool(fac.get("latent_indexed_renoise", False))
        self.renoise_seed = None
        nfpb = int(getattr(self, "num_frame_per_block", 1) or 1)
        k_ns = fac.get("warm_start_frames_nosplit", None)
        self.warm_start_frames_nosplit = None if k_ns is None else int(k_ns)
        if self.warm_start_frames_nosplit is not None:
            if self.warm_start_frames is not None:
                raise ValueError("factorial.warm_start_frames and warm_start_frames_nosplit are "
                                 "exclusive")
            if self.warm_start_frames_nosplit < 1 or self.denoising_step_list_first_chunk is None:
                raise ValueError("factorial.warm_start_frames_nosplit needs K >= 1 and "
                                 "denoising_step_list_first_chunk")
            if self.warm_start_frames_nosplit % nfpb != 0:
                raise ValueError(f"factorial.warm_start_frames_nosplit={k_ns} must be a multiple "
                                 f"of num_frame_per_block={nfpb} (no block splitting)")
        cad = fac.get("echo_cadence_latents", None)
        self.echo_cadence_latents = None if cad is None else int(cad)
        if self.echo_cadence_latents is not None:
            if not self.echo_enabled:
                raise ValueError("factorial.echo_cadence_latents needs model_kwargs.echo_memory")
            if nfpb != 1:
                raise ValueError("factorial.echo_cadence_latents is defined at num_frame_per_block "
                                 f"1 only (got {nfpb}): at k > 1 a per-c update would change "
                                 "within-block causality")
            if self.echo_cadence_latents < 2:
                raise ValueError("factorial.echo_cadence_latents must be >= 2 (1 is the default)")
            c = self.echo_cadence_latents
            budget = (sa.echo_anchor_frames + sa.echo_recent_past) * c + sa.echo_compressed_frames + c
            window = sa.max_attention_size // sa.frame_seq_length
            if budget > window:
                print(f"[factorial][WARNING] Echo frame budget {budget} > local window {window} "
                      f"with cadence {c}: use the chunk-{c} Echo overlay (entry counts)")
            else:
                print(f"[factorial] Echo frame budget {budget}/{window} with cadence {c}")
            for block in self.generator.model.blocks:
                block.self_attn.echo_cadence_latents = c
        pin_src = bool(fac.get("echo_pin_source", False))
        pin_lat = fac.get("echo_pin_latent", None)
        self.echo_pin_latent = None
        if pin_src or pin_lat is not None:
            if pin_src and pin_lat is not None:
                raise ValueError("factorial.echo_pin_source and echo_pin_latent are exclusive")
            if not self.echo_enabled:
                raise ValueError("factorial.echo_pin_* needs model_kwargs.echo_memory")
            p = 0 if pin_src else int(pin_lat)
            if not pin_src and not 1 <= p < int(sa.echo_anchor_pool):
                raise ValueError("factorial.echo_pin_latent: a GENERATED latent id in [1, echo_anchor_pool) "
                                 "(the source is echo_pin_source)")
            if self.echo_cadence_latents is not None or float(getattr(sa, "echo_spectral_strength", 0.0)) > 0:
                raise ValueError("factorial.echo_pin_* is defined on the plain Echo schedule only "
                                 "(not with echo_cadence_latents or the spectral anchor)")
            if int(sa.echo_anchor_frames) < 2:
                raise ValueError("factorial.echo_pin_* needs echo_anchor_frames >= 2 (one slot pinned, "
                                 ">= 1 rotating)")
            if p >= 1 and int(sa.echo_recent_past) < 1:
                raise ValueError("factorial.echo_pin_latent needs a recent store (echo_recent_frames >= 2): "
                                 "otherwise the first generated block would attend no source latent")
            budget = (1 + (sa.echo_anchor_frames - 1) * nfpb + sa.echo_compressed_frames
                      + sa.echo_recent_past * nfpb + nfpb)
            window = sa.max_attention_size // sa.frame_seq_length
            if budget > window:
                raise ValueError(f"factorial.echo_pin_*: max attended frames {budget} > window {window} "
                                 "(the cap would drop the pinned slot)")
            print(f"[factorial] Echo pin: latent {p} ({'source' if p == 0 else 'generated'}) holds one "
                  f"of {sa.echo_anchor_frames} anchor slots for the whole rollout, "
                  f"{sa.echo_anchor_frames - 1} rotate; max attended frames {budget}/{window}")
            self.echo_pin_latent = p
            for block in self.generator.model.blocks:
                block.self_attn.echo_pin_latent = p
        # H2 (opt-in): motion-compensated admission to Echo's long-term compressed memory. At each compression
        # event an evicted token whose temporal support by the next two committed latents (best local cosine of raw
        # values within +-radius / +-2 radius tokens) is a robust low outlier of its latent gets a bounded penalty
        # (<= lam std of the candidate scores) on Echo's calibrated-query admission score; it persists with the token.
        adm = fac.get("echo_admission", None)
        self.echo_admission = None
        if adm:
            adm = adm if hasattr(adm, "get") else {}
            radius, kappa, lam = int(adm.get("radius", 2)), float(adm.get("kappa", 3.0)), float(adm.get("lam", 1.0))
            if not self.echo_enabled:
                raise ValueError("factorial.echo_admission needs model_kwargs.echo_memory")
            if radius < 1 or kappa <= 0 or lam <= 0:
                raise ValueError("factorial.echo_admission: radius >= 1, kappa > 0, lam > 0")
            if self.echo_cadence_latents is not None or getattr(sa, "material_reader", None) is not None:
                raise ValueError("factorial.echo_admission is defined on the plain Echo schedule (no cadence / material reader)")
            self.echo_admission = dict(radius=radius, kappa=kappa, lam=lam)
            for i, block in enumerate(self.generator.model.blocks):
                block.self_attn.echo_admission = dict(self.echo_admission)
                block.self_attn.echo_admission_log = [] if i == 0 else None
            print(f"[factorial] echo_admission radius={radius} kappa={kappa} lam={lam}: temporal-support penalty on Echo's "
                  f"compressed-memory admission score (layer-0 decisions logged)")
        norm = fac.get("ema_alpha_normalize", None)
        self.ema_alpha_normalize = None
        if norm:
            if not bool(sa.ema_sink) or int(sa.sink_size) <= 1:
                raise ValueError("factorial.ema_alpha_normalize needs model_kwargs.ema_sink: true "
                                 "with sink_size > 1 (the EMA soft-sink zone)")
            a_ref = float(norm.get("alpha_ref", sa.compression_alpha))
            k_ref = int(norm.get("k_ref", 3))
            if not 0.0 < a_ref < 1.0 or k_ref < 1:
                raise ValueError("factorial.ema_alpha_normalize: 0 < alpha_ref < 1, k_ref >= 1")
            if abs(a_ref - float(sa.compression_alpha)) > 1e-12:
                raise ValueError(f"factorial.ema_alpha_normalize.alpha_ref={a_ref} must equal "
                                 f"model_kwargs.compression_alpha={sa.compression_alpha} (both are "
                                 "the per-block RETENTION of the reference deployment)")
            a_k = a_ref if nfpb == k_ref else a_ref ** (nfpb / k_ref)
            for block in self.generator.model.blocks:
                block.self_attn.compression_alpha = a_k
            self.generator.model.compression_alpha = a_k
            self.ema_alpha_normalize = dict(alpha_ref=a_ref, k_ref=k_ref, k=nfpb,
                                            compression_alpha_k=a_k, mixing_weight_k=1.0 - a_k)
        # ---- C2 source_schedule, C3 role_routing, DyMoS single-component baseline ----
        self._nominal_regular = [int(t) for t in args.denoising_step_list]
        fc = getattr(args, "denoising_step_list_first_chunk", None)
        self._nominal_first = [int(t) for t in fc] if fc is not None else list(self._nominal_regular)
        # `factorial.regular_step_list` (opt-in): the denoising schedule of EVERY regular
        # block (nominal timesteps, config units), replacing args.denoising_step_list; warped exactly as
        # __init__ warps it (warp_denoising_step); the first-block list and the clean refresh are
        # unchanged. Arm steps4 = [1000, 750, 500, 250]: 4 evaluations per block instead of 2, the
        # sampler factor of the student-vs-native-DyMoS gap isolated inside the student.
        rsl = fac.get("regular_step_list", None)
        self.regular_step_list = None
        if rsl is not None:
            lst = [int(t) for t in (rsl.split(",") if isinstance(rsl, str) else rsl)]
            if not lst or any(not 0 < t <= 1000 for t in lst) or any(a <= b for a, b in zip(lst, lst[1:])):
                raise ValueError("factorial.regular_step_list: a strictly decreasing list of nominal timesteps in "
                                 f"(0, 1000], e.g. [1000, 750, 500, 250] (got {rsl})")
            steps = torch.tensor(lst, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                steps = timesteps[1000 - steps]
            self.denoising_step_list = steps
            self._nominal_regular = lst
            self.regular_step_list = lst
            print(f"[factorial] regular_step_list={lst}: every regular block runs {len(lst)} denoising evaluations "
                  f"(nominal {lst}, model {[round(float(t), 1) for t in steps]}); first-block list "
                  f"{self._nominal_first} and the clean refresh unchanged")
        # `factorial.refine_last: n` (opt-in, arm fgs_refine): n extra denoising evaluations of every
        # REGULAR block at its last nominal timestep before emission (nominal [1000, 500] -> [1000, 500, 500]: the x0 of
        # the t = 500 evaluation is re-noised to t = 500 and evaluated again with the same history read, the last x0 is
        # emitted and refreshed into the cache as usual). Implemented as the step list with the last entry repeated, so
        # the sampler loop is untouched; under latent_indexed_renoise the re-noise draw is keyed by (latent, timestep)
        # and is therefore the SAME eps as the first re-noise to t = 500 (a deterministic fixed-point refinement, not a
        # fresh sample). First block and clean refresh unchanged.
        rfl = fac.get("refine_last", None)
        self.refine_last = None
        if rfl not in (None, False):
            n_ref = int(rfl)
            if not 1 <= n_ref <= 3:
                raise ValueError(f"factorial.refine_last: 1..3 extra evaluations at the last nominal timestep (got {rfl})")
            self.denoising_step_list = torch.cat([self.denoising_step_list,
                                                  self.denoising_step_list[-1:].repeat(n_ref)])
            self._nominal_regular = list(self._nominal_regular) + [self._nominal_regular[-1]] * n_ref
            self.refine_last = n_ref
            print(f"[factorial] refine_last={n_ref}: every regular block runs {len(self._nominal_regular)} denoising "
                  f"evaluations (nominal {self._nominal_regular}, model "
                  f"{[round(float(t), 1) for t in self.denoising_step_list]}); the extra evaluation(s) re-noise the last "
                  f"x0 to the last timestep (latent-indexed: the same eps) and re-evaluate before emission; first block "
                  f"and clean refresh unchanged")
        p1 = getattr(args, "phase1", {}) or {}
        ss = fac.get("source_schedule", None)
        self.source_schedule = None
        if ss:
            mode = str(ss.get("mode", "lowpass"))
            if mode not in ("lowpass", "downweight"):
                raise ValueError("factorial.source_schedule.mode must be lowpass | downweight")
            sigma = float(ss.get("sigma", 1.0))
            weight = float(ss.get("weight", 0.5))
            if mode == "lowpass" and not sigma > 0:
                raise ValueError("factorial.source_schedule.sigma (latent px) must be > 0 for lowpass")
            if mode == "downweight" and not 0.0 <= weight <= 1.0:
                raise ValueError("factorial.source_schedule.weight must be in [0, 1] (1.0 == sharp: "
                                 "identity check only)")
            steps = str(ss.get("steps", "by_noise"))
            if steps not in ("by_noise", "last_sharp", "always_soft"):
                raise ValueError("factorial.source_schedule.steps must be by_noise | last_sharp | always_soft")
            t_switch = ss.get("t_switch", None)
            t_switch = (min(self._nominal_regular) + 1) if t_switch is None else int(t_switch)
            if not self._relativistic_rope:
                raise ValueError("factorial.source_schedule needs model_kwargs.relativistic_rope "
                                 "(raw-K cache with the frame-0 hard anchor, or Echo)")
            if not self.echo_enabled and (int(sa.sink_size) < 1 or self.pure_sliding_window):
                raise ValueError("factorial.source_schedule: this policy holds no source entry "
                                 "(sliding window, sink_size 0): nothing to schedule")
            if self.alg_enabled or float((p1.get("dymos", {}) or {}).get("gamma", 0.0)):
                raise ValueError("factorial.source_schedule is exclusive with the Phase-1 ALG / "
                                 "DyMoS ports (phase1.alg / phase1.dymos write or bias the same slot)")
            if self.echo_cadence_latents is not None:
                raise ValueError("factorial.source_schedule is not defined with echo_cadence_latents")
            self.source_schedule = dict(mode=mode, sigma=sigma, weight=weight, steps=steps,
                                        t_switch=t_switch)
            self._source_table = {}
            for name, lst in (("first", self._nominal_first), ("regular", self._nominal_regular)):
                self._source_table[name] = [
                    (t, "soft" if self._source_reads_soft(lst, i) else "sharp") for i, t in enumerate(lst)]
            print(f"[factorial] source_schedule mode={mode} sigma={sigma} weight={weight} steps={steps} "
                  f"t_switch={t_switch}: first block "
                  + " ".join(f"{t}->{r}" for t, r in self._source_table["first"]) + "; regular "
                  + " ".join(f"{t}->{r}" for t, r in self._source_table["regular"])
                  + "; refresh(t=0)->sharp; history/commits: sharp only")
        # per-evaluation schedule record: the NOMINAL config timestep, the WARPED
        # timestep the model receives (denoising_step_list after warp_denoising_step; nominal
        # [1000, 500] -> model [1000, 833] at shift 5) and the scheduler's sigma / alpha at that
        # evaluation (x_t = alpha * x0 + sigma * eps, FlowMatchScheduler.add_noise), plus the entry
        # read. Written per video into <video>_context.json so the 2-step adaptation of the "late
        # stage" is stated honestly: the last evaluation sits at sigma 0.833, not near 0.
        self.schedule_record = {}
        for name, nominal, model_t in (("first", self._nominal_first, self.denoising_step_list_first_chunk),
                                       ("regular", self._nominal_regular, self.denoising_step_list)):
            if model_t is None:
                model_t = self.denoising_step_list
            rows = []
            for i, t_nom in enumerate(nominal):
                t_mod = float(model_t[i])
                sig = self._sigma_of(t_mod)
                read = ("soft" if self._source_reads_soft(nominal, i) else "sharp") if self.source_schedule else "sharp"
                rows.append(dict(index=i, t_nominal=int(t_nom), t_model=round(t_mod, 3), sigma=round(sig, 6),
                                 alpha=round(1.0 - sig, 6), entry=read))
            rows.append(dict(index=None, t_nominal=0, t_model=0.0, sigma=0.0, alpha=1.0, entry="sharp",
                             note="clean refresh (context_noise 0); commits sharp"))
            self.schedule_record[name] = rows
        self.schedule_record["steps"] = self.source_schedule["steps"] if self.source_schedule else None
        self.schedule_record["t_switch"] = self.source_schedule["t_switch"] if self.source_schedule else None
        self.schedule_record["entries"] = "soft/sharp" if self.source_schedule else "sharp only (one source entry)"
        rr = fac.get("role_routing", None)
        self.role_routing = None
        if rr:
            if self.source_schedule is None:
                raise ValueError("factorial.role_routing routes between the two source entries of "
                                 "factorial.source_schedule: set that first")
            if str(rr.get("source", "volatility")) != "volatility":
                raise ValueError("factorial.role_routing.source must be 'volatility' (the student's "
                                 "own clean-latent temporal variance; no other source exists)")
            warm_s = float(rr.get("warmup_s", 1.5))
            fps = float(rr.get("latent_fps", 4.0))
            if not warm_s > 0 or not fps > 0:
                raise ValueError("factorial.role_routing.warmup_s and latent_fps must be > 0")
            thr = rr.get("threshold", "auto")
            if thr != "auto":
                thr = float(thr)
                if not 0.0 < thr < 1.0:
                    raise ValueError("factorial.role_routing.threshold: 'auto' (Otsu on the log "
                                     "variance) or a quantile in (0, 1) (tokens above it are material)")
            # circularity guard: no mask of any kind may be configured
            self._assert_no_mask_config(args)
            warm_latents = max(1, int(round(warm_s * fps)))
            shuffle = bool(rr.get("shuffle", False))
            unc = rr.get("uncertain", None)
            if unc is None or unc is False:
                band = 0.0
            elif unc == "band":
                band = ROLE_UNCERTAIN_BAND_LOG10
            else:
                band = float(unc)
                if band < 0:
                    raise ValueError("factorial.role_routing.uncertain: a half-width >= 0 in log10-variance "
                                     "units, or 'band' (the pre-registered ROLE_UNCERTAIN_BAND_LOG10)")
            self.role_routing = dict(source="volatility", warmup_s=warm_s, latent_fps=fps,
                                     warmup_latents=warm_latents, threshold=thr, shuffle=shuffle,
                                     uncertain_band=band)
            print(f"[factorial] role_routing source=volatility warmup_s={warm_s} ({warm_latents} "
                  f"latents) threshold={thr} uncertain_band={band} shuffle={shuffle}: online map of the "
                  f"student's own clean-latent temporal variation, frozen after warm-up; C3 routes "
                  f"QUERIES between two complete source entries: low-variation tokens read the sharp "
                  f"entry at every step, high-variation tokens (and tokens in the uncertain band) the "
                  f"scheduled one; before the map every query follows the schedule; evaluation masks "
                  f"never read" + ("; CONTROL: the map is spatially shuffled (same class proportions, "
                                   "permutation seeded by the video seed)" if shuffle else ""))
        dy = fac.get("dymos_ref_scale", None)
        self.dymos_ref_scale = None
        if dy:
            scale = float(dy.get("scale", 0.6))
            early = int(dy.get("early_steps", 1))
            if not scale > 0:
                raise ValueError("factorial.dymos_ref_scale.scale (paper gamma) must be > 0")
            if early < 1:
                raise ValueError("factorial.dymos_ref_scale.early_steps must be >= 1")
            if float((p1.get("dymos", {}) or {}).get("gamma", 0.0)):
                raise ValueError("factorial.dymos_ref_scale is exclusive with phase1.dymos.gamma "
                                 "(the Phase-1 port biases the same keys)")
            if not self._relativistic_rope or (not self.echo_enabled and (
                    int(sa.sink_size) < 1 or self.pure_sliding_window)):
                raise ValueError("factorial.dymos_ref_scale needs a policy with a source entry "
                                 "(sink_size >= 1 or Echo) on the relativistic-RoPE path")
            self.dymos_ref_scale = dict(scale=scale, early_steps=early)
            for block in self.generator.model.blocks:
                block.self_attn.dymos_ref_scale = scale
                block.self_attn.dymos_active = False
            print(f"[factorial] dymos_ref_scale gamma={scale}: additive pre-softmax bias -gamma on the "
                  f"source-key logits of every generated-frame query (uniform phi=1, all self-attention "
                  f"layers; arXiv 2605.19398 Eq. 6) during the first {early} evaluation(s) of every "
                  f"block; off at the clean refresh and at the source pass")
        # ---- training-free ports: packcache_decay (PackCache decay budget) / fp_pool
        # (FramePack-schedule K/V pooling); read-time only, exclusive with each other and with C2 / C3 /
        # DyMoS (single-mechanism baseline rows); on Echo they need pinsrc (the source must be attended
        # at every block, PackCache's condition anchoring / FramePack's clean first-frame input).
        pc = fac.get("packcache_decay", None)
        fpp = fac.get("fp_pool", None)
        pc = None if pc is False else pc                  # an empty mapping {} means "on, all defaults"
        fpp = None if fpp is False else fpp
        self.packcache_decay = self.fp_pool = None
        if pc is not None or fpp is not None:
            if pc is not None and fpp is not None:
                raise ValueError("factorial.packcache_decay and fp_pool are exclusive")
            name = "packcache_decay" if pc is not None else "fp_pool"
            pc = pc or {}
            fpp = fpp or {}
            # factorial.compose: packcache_decay (source anchored) may run WITH C2 (source_schedule) and
            # the DyMoS port - they act on the source entry, which the decay never thins; everything else stays exclusive
            composable = bool(fac.get("compose", False)) and name == "packcache_decay" \
                and str(pc.get("source", "full")) == "full" and self.role_routing is None
            if (self.source_schedule is not None or self.role_routing is not None or self.dymos_ref_scale is not None) \
                    and not composable:
                raise ValueError(f"factorial.{name} is a single-mechanism port: not with source_schedule / "
                                 "role_routing / dymos_ref_scale (factorial.compose allows packcache_decay with a "
                                 "full source + source_schedule / dymos_ref_scale)")
            if self.alg_enabled or float((p1.get("dymos", {}) or {}).get("gamma", 0.0)) \
                    or float(getattr(sa, "echo_spectral_strength", 0.0)) > 0 or getattr(sa, "material_reader", None) is not None:
                raise ValueError(f"factorial.{name} is not defined with the Phase-1 ALG / DyMoS ports, the "
                                 "spectral anchor or a material reader")
            if not self._relativistic_rope:
                raise ValueError(f"factorial.{name} needs model_kwargs.relativistic_rope")
            if self.echo_cadence_latents is not None:
                raise ValueError(f"factorial.{name} is not defined with echo_cadence_latents")
            if self.echo_enabled:
                if getattr(self, "echo_pin_latent", None) != 0:
                    raise ValueError(f"factorial.{name} on Echo needs factorial.echo_pin_source (the source "
                                     "frame is the condition anchor / clean first-frame input at every block)")
            elif int(sa.sink_size) < 1 or self.pure_sliding_window:
                raise ValueError(f"factorial.{name}: the sliding window holds no source entry (sink_size 0)")
            fs = int(sa.frame_seq_length)
            W = int(sa.max_attention_size // fs)
            if name == "packcache_decay":
                rho = float(pc.get("rho", 0.5))
                rule = str(pc.get("rule", "keynorm"))
                if not 0.0 < rho < 1.0:
                    raise ValueError("factorial.packcache_decay.rho must be in (0, 1) (paper: 1/2)")
                if rule not in ("keynorm", "stride", "pool"):
                    raise ValueError("factorial.packcache_decay.rule must be keynorm | stride (our pre-registered "
                                     "intra-frame rules; the paper states none) | pool multi-resolution decay)")
                self.packcache_decay = dict(rho=rho, rule=rule, window=W)
                budgets = [int(fs * rho ** min(d, W - 1) + 0.5) for d in range(1, W)]
                self.packcache_decay["budget_tokens_by_distance"] = budgets
                for block in self.generator.model.blocks:
                    block.self_attn.packcache_decay = dict(rho=rho, rule=rule)
                    block.self_attn.pack_stats = []
                src = str(pc.get("source", "full"))
                if src not in ("full", "decay"):
                    raise ValueError("factorial.packcache_decay.source must be full | decay")
                self.packcache_decay["source"] = src
                # prev_floor (opt-in): the newest history latent (id cur - 1, the latent generated just
                # before the block) keeps at least round(floor * 1560) tokens by fixed spatial stride; older latents keep
                # the rho budget and rule: the flashes come from the thinned newest history latent)
                pfl = pc.get("prev_floor", None)
                if pfl is not None:
                    pfl = float(pfl)
                    if not 0.0 < pfl <= 1.0:
                        raise ValueError("factorial.packcache_decay.prev_floor must be in (0, 1]")
                    if src != "full":
                        raise ValueError("factorial.packcache_decay.prev_floor needs the source anchored (source: full)")
                    if rule == "pool":
                        raise ValueError("factorial.packcache_decay.prev_floor is not defined with rule pool")
                    self.packcache_decay["prev_floor"] = pfl
                    for block in self.generator.model.blocks:
                        block.self_attn.packcache_decay["prev_floor"] = pfl
                    print(f"[factorial] packcache_decay prev_floor={pfl}: the newest history latent keeps max(round(1560 * "
                          f"{rho}), {int(fs * pfl + 0.5)}) tokens by fixed spatial stride; older latents unchanged")
                # prev_mode (opt-in): pool2 = the newest history latent 2x2-pooled (full spatial coverage, 390
                # tokens); soft = the newest latent whole with the frame logit bias log(budget / 1560)
                pmode = pc.get("prev_mode", None)
                if pmode is not None:
                    pmode = str(pmode)
                    if pmode not in ("pool2", "soft", "pool2_stride"):
                        raise ValueError("factorial.packcache_decay.prev_mode must be pool2 | soft | pool2_stride")
                    if rule == "pool":
                        raise ValueError("factorial.packcache_decay.prev_mode is not defined with rule pool")
                    if src != "full" or pfl is not None:
                        raise ValueError("factorial.packcache_decay.prev_mode needs source: full and no prev_floor")
                    if pmode == "soft" and bool(fac.get("compose", False)):
                        raise ValueError("factorial.packcache_decay.prev_mode soft is not composed with C2 / DyMoS")
                    self.packcache_decay["prev_mode"] = pmode
                    for block in self.generator.model.blocks:
                        block.self_attn.packcache_decay["prev_mode"] = pmode
                    print(f"[factorial] packcache_decay prev_mode={pmode}: the newest history latent is "
                          + {"pool2": "2x2 average-pooled (390 of 1560 tokens, every grid cell covered)",
                             "pool2_stride": f"its 2x2-pooled summary (390) plus {int(fs * rho + 0.5)} exact tokens by stride",
                             "soft": f"kept whole with frame logit bias log({int(fs * rho + 0.5)}/{fs}) (no holes, budget mass)"}[pmode]
                          + "; older latents unchanged")
                if rule == "pool":
                    if src != "full":
                        raise ValueError("factorial.packcache_decay.rule pool needs source: full")
                    print("[factorial] packcache_decay rule=pool multi-resolution decay): a history latent at age d "
                          "is 2^min(d,5) x 2^min(d,5) average-pooled (390, 104, 28, 8, 2, ... tokens), nothing dropped; "
                          "rho only labels the budget log line")
                if src == "decay":
                    # fg_nosrc: no condition anchor - the source frame is decayed like any history frame
                    # at its true age (latent distance cur - 0), so it is dropped once rho^d * 1560 rounds to 0
                    for block in self.generator.model.blocks:
                        block.self_attn.pack_source_decay = True
                    print(f"[factorial] packcache_decay source=decay: the SOURCE frame is decayed at its true age (no anchor)")
                print(f"[factorial] packcache_decay rho={rho} rule={rule} W={W}: PackCache cross-frame decay budget "
                      f"b_d = rho^min(d, W-1) per history frame at latent distance d (source frame and current block "
                      f"full; frames with budget 0 dropped); tokens per frame d=1..{W - 1}: {budgets} of {fs}; "
                      f"intra-frame rule '{rule}' is OUR choice (the paper states none); read-time only")
            else:
                full, half, quarter = int(fpp.get("full", 1)), int(fpp.get("half", 2)), int(fpp.get("quarter", 16))
                beyond = str(fpp.get("beyond", "drop"))
                if min(full, half, quarter) < 0 or beyond not in ("drop", "quarter"):
                    raise ValueError("factorial.fp_pool: full / half / quarter >= 0 and beyond drop | quarter")
                self.fp_pool = dict(full=full, half=half, quarter=quarter, beyond=beyond)
                for block in self.generator.model.blocks:
                    block.self_attn.fp_pool = dict(self.fp_pool)
                    block.self_attn.pack_stats = []
                print(f"[factorial] fp_pool full={full} half={half} quarter={quarter} beyond={beyond}: FramePack-schedule "
                      f"K/V pooling (1x / 2x2 / 4x4 average pooling of the ROPED keys and the values by latent distance; "
                      f"source frame and current block 1x; {'older frames dropped' if beyond == 'drop' else 'older frames 4x4'}); "
                      f"an analogue of the paper's learned per-rate patchify kernels, not FramePack; read-time only")
            # dual-sink base (ema_sink): the EMA soft-sink slots 1..s-1 carry no latent id once
            # an eviction has blended them; `full` (default) keeps them whole like the hard sink (they are the
            # base's memory), `budget` treats them as the oldest history (decay: minimum budget; pool: beyond)
            soft = str(fac.get("pack_soft_sink", "full"))
            if soft not in ("full", "budget"):
                raise ValueError("factorial.pack_soft_sink must be full | budget")
            self.pack_soft_sink = None
            if not self.echo_enabled and bool(sa.ema_sink) and int(sa.sink_size) > 1:
                self.pack_soft_sink = soft
                for block in self.generator.model.blocks:
                    block.self_attn.pack_soft_sink = soft
                print(f"[factorial] {name} on the dual sink: hard sink (source) full, EMA soft-sink slots {soft}")
        # ---- /: MATERIAL-AWARE FORGETTING (read-time; nothing written to the cache) --------
        # forget {roles: online_var | none, rho: 0.5, rho_material: 0.3, rho_rigid: 0.7, recent_s: 0.75, warmup_s: 1.5,
        #         window_s: 1.5, map_refresh_s: 5.0, shuffle: false, rule: keynorm | stride, threshold: auto | <q>}
        # Per attended history frame at age d latents (d * 0.25 s; capped at W - 1 like the uniform budget) the frame
        # keeps the UNIFORM budget B = round(1560 * rho^d) tokens (the packcache_decay budget), or all of them inside
        # the recent window (frames newer than recent_s, counted from the newest latent; the current block, the
        # source and frames without a latent id follow the packcache rules: full / full / budget at d = W - 1).
        # Roles (material = high temporal variation, rigid = low) from the C3 estimator on the student's OWN clean
        # latents: first frozen after warmup_s, then re-estimated every map_refresh_s of video time from the last
        # window_s of clean latents (evaluation masks are never read). B is split between the roles by the equal-budget
        # solver (causal_model._forget_split): K_role proportional to N_role * rho_role0^d, K_m + K_r = B exactly.
        # Before the first map (and with roles: none) the frame's B tokens are chosen over all tokens.
        fgc = fac.get("forget", None)
        self.forget = None
        if fgc:
            on = dict(source_schedule=self.source_schedule, role_routing=self.role_routing,
                      dymos_ref_scale=self.dymos_ref_scale, packcache_decay=self.packcache_decay, fp_pool=self.fp_pool)
            if bool(fac.get("compose", False)):
                on.pop("dymos_ref_scale")        #: the source is never thinned under forget -> DyMoS composes
            if [k for k, v in on.items() if v is not None] or self.echo_cadence_latents is not None or self.alg_enabled \
                    or float((p1.get("dymos", {}) or {}).get("gamma", 0.0)) \
                    or float(getattr(sa, "echo_spectral_strength", 0.0)) > 0 or getattr(sa, "material_reader", None) is not None:
                raise ValueError("factorial.forget is a single-mechanism memory policy: not with C2 / C3 / DyMoS / "
                                 "packcache_decay / fp_pool / cadence / spectral / Phase-1 ports")
            if not self._relativistic_rope:
                raise ValueError("factorial.forget needs model_kwargs.relativistic_rope")
            if self.echo_enabled:
                if getattr(self, "echo_pin_latent", None) != 0:
                    raise ValueError("factorial.forget on Echo needs factorial.echo_pin_source (the source is attended "
                                     "at every block)")
            elif int(sa.sink_size) < 1 or self.pure_sliding_window:
                raise ValueError("factorial.forget: the sliding window holds no source entry")
            roles = str(fgc.get("roles", "online_var"))
            if roles not in ("online_var", "none", "srcred"):
                raise ValueError("factorial.forget.roles must be online_var | none | srcred")
            # roles srcred (source-complementary memory): no role map; within each history frame's
            # budget keep the tokens LEAST (redundancy: least) or MOST (most, control) redundant with the source, where
            # redundancy(p, f) = mean over heads of cos(V_f[p], V_source[p]) - raw values at the same grid position
            redundancy = str(fgc.get("redundancy", "least"))
            if roles == "srcred" and redundancy not in ("least", "most"):
                raise ValueError("factorial.forget.redundancy must be least | most")
            # controls: the REFERENCE of the redundancy score (same budget, selection and window):
            #   source   this scene's source V at the same grid position (§17.13, the default)
            #   adjacent the next-newer frame in the attended window (temporal redundancy, Forcing-KV-like); for the newest
            #            budgeted history frame that is the oldest frame of the recent window / current block
            #   xref     the source V of a DIFFERENT dev scene (fixed derangement, captured once per video by a scratch
            #            prefix pass; never written to the real cache; the conditioning stays this scene's)
            #   pshuf    this scene's source V with its token positions permuted (seeded by the video seed)
            reference = str(fgc.get("reference", "source"))
            if reference not in ("source", "adjacent", "xref", "pshuf"):
                raise ValueError("factorial.forget.reference must be source | adjacent | xref | pshuf")
            if reference != "source" and roles != "srcred":
                raise ValueError("factorial.forget.reference applies to roles: srcred only")
            rho = float(fgc.get("rho", 0.5))
            rm, rr = float(fgc.get("rho_material", 0.3)), float(fgc.get("rho_rigid", 0.7))
            recent_s, warm_s = float(fgc.get("recent_s", 0.75)), float(fgc.get("warmup_s", 1.5))
            win_s, refresh_s = float(fgc.get("window_s", warm_s)), float(fgc.get("map_refresh_s", 5.0))
            rule = str(fgc.get("rule", "keynorm"))
            thr = fgc.get("threshold", "auto")
            # (opt-in; absent = off, every earlier arm unchanged):
            #   sticky  temporally stable selection: a history latent's kept tokens are ranked ONCE, when it is first
            #           attended with a partial budget (key norm for roles none; source similarity, ties by key norm, for
            #           roles srcred with the source reference), and afterwards keep the first B' <= B of that ranking -
            #           the kept set only shrinks (nested over time); same per-frame budget as without sticky
            #   soft    soft forgetting: keep EVERY token of the attended history and add the per-frame logit bias
            #           d * log(rho) (0 for the source, the current block and the recent window); roles none only;
            #           rho in (0, 1] (rho = 1: zero bias, the unchanged default window and kernel)
            sticky, soft_fg = bool(fgc.get("sticky", False)), bool(fgc.get("soft", False))
            if sticky and soft_fg:
                raise ValueError("factorial.forget.sticky and soft are exclusive")
            if sticky and not (rule == "keynorm" and (roles == "none" or (roles == "srcred" and reference == "source"))):
                raise ValueError("factorial.forget.sticky needs a time-invariant ranking: roles none with rule keynorm, or "
                                 "roles srcred with reference source (the adjacent / xref / pshuf references and the "
                                 "online role map change between blocks)")
            if soft_fg:
                if roles != "none":
                    raise ValueError("factorial.forget.soft keeps every token (no selection): roles none only")
                if not 0 < rho <= 1:
                    raise ValueError("factorial.forget.soft: rho in (0, 1] (bias d * log(rho); rho = 1 -> no bias)")
                if bool(fac.get("compose", False)):
                    raise ValueError("factorial.forget.soft is not composed with C2 / DyMoS (one biased kernel per read)")
            if soft_fg and rho == 1.0:
                pass                                                  # zero bias: the default window (tested bitwise)
            elif not (0 <= rho < 1 and 0 < rm < 1 and 0 < rr < 1):
                #: rho = 0 drops every history frame outside the recent window (budget round(1560 * 0^d)
                # = 0 for d >= 1); the source, the recent window and the current block stay full
                raise ValueError("factorial.forget: rho in [0, 1), rho_material and rho_rigid in (0, 1)")
            if recent_s < 0 or warm_s <= 0 or win_s <= 0 or refresh_s <= 0:
                raise ValueError("factorial.forget: recent_s >= 0, warmup_s / window_s / map_refresh_s > 0")
            if rule not in ("keynorm", "stride"):
                raise ValueError("factorial.forget.rule must be keynorm | stride")
            if thr != "auto":
                thr = float(thr)
                if not 0.0 < thr < 1.0:
                    raise ValueError("factorial.forget.threshold: 'auto' (Otsu) or a quantile in (0, 1)")
            if roles == "online_var":
                self._assert_no_mask_config(args)                     # circularity guard
            lat_s = 0.25                                              # one latent = 4 frames at 16 fps
            self.forget = dict(roles=roles, rho=rho, rho_material=rm, rho_rigid=rr, recent_s=recent_s,
                               recent_latents=int(round(recent_s / lat_s)), warmup_s=warm_s,
                               warmup_latents=max(1, int(round(warm_s / lat_s))), window_s=win_s,
                               window_latents=max(2, int(round(win_s / lat_s))), map_refresh_s=refresh_s,
                               shuffle=bool(fgc.get("shuffle", False)), rule=rule, threshold=thr, latent_s=lat_s)
            if roles == "srcred":
                self.forget["redundancy"] = redundancy
                if reference != "source":
                    self.forget["reference"] = reference
            if sticky:
                self.forget["sticky"] = True
            if soft_fg:
                self.forget["soft"] = True
            # prev_floor (opt-in): the newest history latent (id cur - 1) keeps max(B, round(floor * 1560))
            # tokens by FIXED SPATIAL STRIDE (uniform coverage); older latents keep this path's selection
            fg_pfl = fgc.get("prev_floor", None)
            if fg_pfl is not None:
                fg_pfl = float(fg_pfl)
                if not 0.0 < fg_pfl <= 1.0:
                    raise ValueError("factorial.forget.prev_floor must be in (0, 1]")
                if sticky or soft_fg or roles not in ("none", "srcred") or recent_s != 0.0:
                    raise ValueError("factorial.forget.prev_floor: roles none | srcred, recent_s 0, not sticky / soft")
                self.forget["prev_floor"] = fg_pfl
            # stratify (opt-in): every partial-budget history latent keeps one token per raster span (B spans),
            # the best of the path's ranking (srcred: most / least source-redundant; roles none: key norm)
            # round 5 prev_mode (opt-in): the newest history latent as its 2x2-pooled summary (pool2) or that
            # summary plus its round(1560 rho) exact tokens by stride (pool2_stride); older latents keep this path's selection
            fg_pmode = fgc.get("prev_mode", None)
            if fg_pmode is not None:
                fg_pmode = str(fg_pmode)
                if fg_pmode not in ("pool2", "pool2_stride"):
                    raise ValueError("factorial.forget.prev_mode must be pool2 | pool2_stride")
                if fg_pfl is not None or sticky or soft_fg or roles not in ("none", "srcred") or recent_s != 0.0:
                    raise ValueError("factorial.forget.prev_mode: roles none | srcred, recent_s 0, no prev_floor / sticky / soft")
                self.forget["prev_mode"] = fg_pmode
            # H1 (opt-in): prev_split = motion-aware read of the newest history latent from the online
            # volatility map (roles online_var): moving / uncertain tokens exact, static tokens prev_static_frac by
            # stride; role_older false keeps the older-history selection role-agnostic (roles-none rule)
            fg_split = bool(fgc.get("prev_split", False))
            fg_role_older = bool(fgc.get("role_older", True))
            if fg_split or not fg_role_older:
                if roles != "online_var":
                    raise ValueError("factorial.forget.prev_split / role_older need roles online_var (the online map)")
                if fgc.get("prev_freeze") and not fg_split:
                    raise ValueError("factorial.forget.prev_freeze needs prev_split")
                if recent_s != 0.0 or sticky or soft_fg or fg_pfl is not None or fgc.get("prev_mode") is not None \
                        or bool(fgc.get("stratify", False)):
                    raise ValueError("factorial.forget.prev_split: recent_s 0, no sticky / soft / prev_floor / prev_mode / stratify")
            if fg_split:
                psf, band = float(fgc.get("prev_static_frac", 0.3)), float(fgc.get("prev_band", 0.15))
                if not 0.0 < psf <= 1.0 or band < 0:
                    raise ValueError("factorial.forget.prev_static_frac in (0, 1], prev_band >= 0")
                self.forget.update(prev_split=True, prev_static_frac=psf, prev_band=band)
                # controls: prev_select map (default) | inverted | random (budget-matched); source full | decay
                psel = str(fgc.get("prev_select", "map"))
                if psel not in ("map", "inverted", "random"):
                    raise ValueError("factorial.forget.prev_select must be map | inverted | random")
                if psel != "map":
                    self.forget["prev_select"] = psel
                fsrc = str(fgc.get("source", "full"))
                if fsrc not in ("full", "decay"):
                    raise ValueError("factorial.forget.source must be full | decay")
                if fsrc == "decay":
                    self.forget["source"] = "decay"
                    print("[factorial] forget source=decay no-anchor control): the SOURCE latent is budgeted at its "
                          "true age like any history latent (dropped once round(1560 * rho^age) is 0)")
                # (opt-in): freeze the motion-aware read's dense mask at the FIRST online map. For a fixed camera
                # the moving regions are a scene property; refreshing from the student's own drifting output lets drift
                # raise variance, relabel the background as moving and copy it exactly (positive feedback,
                if bool(fgc.get("prev_freeze", False)):
                    self.forget["prev_freeze"] = True
            if not fg_role_older:
                self.forget["role_older"] = False
            #: separate opt-in arms; absent keys preserve the locked H1-FM path.
            static_keys = ("prev_query_alpha", "prev_tether", "prev_tether_band")
            static_opts = {key: float(fgc[key]) for key in static_keys if fgc.get(key) is not None}
            if static_opts:
                if any(not math.isfinite(x) or not 0 <= x <= 1 for x in static_opts.values()):
                    raise ValueError("static-fidelity strengths must be finite and in [0,1]")
                if not fg_split or not fgc.get("prev_freeze") or band != 0 or fg_role_older \
                        or self.num_frame_per_block != 1 or psel != "map" or fsrc != "full":
                    raise ValueError("static fidelity requires frozen no-band H1-FM at k=1, source full")
                if not getattr(self.generator.model.blocks[0].self_attn, "echo_memory", False):
                    raise ValueError("static-fidelity trial requires the Echo host")
                if "prev_query_alpha" in static_opts and "prev_tether" in static_opts:
                    raise ValueError("query and value designs are separate trial arms")
                if "prev_tether_band" in static_opts and "prev_tether" not in static_opts:
                    raise ValueError("prev_tether_band requires prev_tether")
                self.forget.update(static_opts)
            fg_strat = bool(fgc.get("stratify", False))
            if fg_strat:
                if sticky or soft_fg or roles not in ("none", "srcred") or (roles == "srcred" and reference != "source"):
                    raise ValueError("factorial.forget.stratify: roles none | srcred (reference source), not sticky / soft")
                self.forget["stratify"] = True
            for block in self.generator.model.blocks:
                block.self_attn.forget = dict(rho=rho, rho_material=rm, rho_rigid=rr,
                                              recent_latents=self.forget["recent_latents"], rule=rule)
                if roles == "srcred":
                    block.self_attn.forget["redundancy"] = redundancy
                    if reference != "source":
                        block.self_attn.forget["reference"] = reference
                if sticky:
                    block.self_attn.forget["sticky"] = True
                if soft_fg:
                    block.self_attn.forget["soft"] = True
                if fg_pfl is not None:
                    block.self_attn.forget["prev_floor"] = fg_pfl
                if fg_strat:
                    block.self_attn.forget["stratify"] = True
                if fg_pmode is not None:
                    block.self_attn.forget["prev_mode"] = fg_pmode
                if fg_split:
                    block.self_attn.forget.update(prev_split=True, prev_static_frac=psf)
                    if self.forget.get("prev_select"):
                        block.self_attn.forget["prev_select"] = self.forget["prev_select"]
                    if self.forget.get("source") == "decay":
                        block.self_attn.pack_source_decay = True
                if not fg_role_older:
                    block.self_attn.forget["role_older"] = False
                if static_opts:
                    block.self_attn.forget.update(static_opts)
                    block.self_attn.forget_static_conf = None
                block.self_attn.forget_prev_dense = None
                block.self_attn.forget_role = None
                block.self_attn.forget_map_version = 0
                block.self_attn.pack_stats = []
            if roles == "srcred":
                print(f"[factorial] forget roles=srcred: per history frame keep the {redundancy.upper()} source-redundant "
                      f"tokens of the uniform budget round(1560 * {rho}^d) (redundancy = mean over heads of cos(V, V_source) "
                      f"at the same grid position, raw values; ties by key norm); no role map; source, current block and "
                      f"recent {recent_s} s full; read-time only")
            if sticky:
                print(f"[factorial] forget sticky: each history latent's kept tokens are ranked once ("
                      f"{'source similarity, ties by key norm' if roles == 'srcred' else 'key norm'}) when it is first "
                      f"attended with a partial budget; later blocks keep the first round(1560 * {rho}^d) of that ranking "
                      f"(nested over time, same budget); read-time only")
            if soft_fg:
                print(f"[factorial] forget SOFT: every history token kept; additive logit bias d * log({rho}) = "
                      f"{math.log(rho) if rho > 0 else float('-inf'):.4f} * d per history frame at age d latents (source, "
                      f"current block, recent {recent_s} s unbiased): frame mass x {rho}^d, within-frame weights unchanged; "
                      f"full-window biased attention (no flash kernel); read-time only")
            print(f"[factorial] forget roles={roles} rho={rho} rho_material0={rm} rho_rigid0={rr} recent {recent_s} s "
                  f"({self.forget['recent_latents']} latents) map: first {warm_s} s, then every {refresh_s} s from the last "
                  f"{win_s} s of the student's own clean latents{' SHUFFLED (seeded)' if self.forget['shuffle'] else ''}; "
                  f"rule {rule}; per history frame the uniform budget round(1560 * {rho}^d) is split K_m : K_r = "
                  f"N_m {rm}^d : N_r {rr}^d (equal total); source and recent window full; read-time only")
        # ---- related KV-memory baselines (training-free ports; single-mechanism rows; default off) ----------------
        # forcing_kv   {head_profile: <json path | mapping with 'layers'>, segments: 6, retention: 0.33, candidates: 3,
        #               score_layer: 1}  Forcing-KV (arXiv 2605.09681) as released (causal_model._fkv_attention)
        # future_kv    {budget: uni | rho03, lam: 0.5, horizon: 6, tau: 0.95, w: 3, merge: true}  Future Forcing (arXiv
        #               2605.30083) at the attended-token budget of fg_uni (rho 0.5, 0.75 s recent window) or fg_rho03
        #               (rho 0.3, no recent window) (causal_model._ffk_attention)
        # tether_cache {memory: 14, recent: 4, alpha: 0.35, tau: 0.6, temperature: 1.0}  TetherCache (arXiv 2606.13035) on
        #               the static sink (causal_model._tc_attention)
        self.forcing_kv = self.future_kv = self.tether_cache = None
        ports = {k_: fac.get(k_, None) for k_ in ("forcing_kv", "future_kv", "tether_cache")}
        ports = {k_: ({} if v_ is True else v_) for k_, v_ in ports.items() if v_ not in (None, False)}
        if ports:
            if len(ports) > 1:
                raise ValueError("factorial.forcing_kv / future_kv / tether_cache are exclusive")
            name = next(iter(ports))
            pcfg = dict(ports[name] or {})
            on = dict(source_schedule=self.source_schedule, role_routing=self.role_routing,
                      dymos_ref_scale=self.dymos_ref_scale, packcache_decay=self.packcache_decay, fp_pool=self.fp_pool,
                      forget=self.forget)
            if [k_ for k_, v_ in on.items() if v_ is not None] or self.echo_cadence_latents is not None \
                    or self.alg_enabled or float((p1.get("dymos", {}) or {}).get("gamma", 0.0)) \
                    or float(getattr(sa, "echo_spectral_strength", 0.0)) > 0 \
                    or getattr(sa, "material_reader", None) is not None \
                    or any(fac.get(k_, None) for k_ in ("local_layer_sizes", "clean_sigma", "svd_vo", "guidance", "compose")):
                raise ValueError(f"factorial.{name} is a single-mechanism port: not with C2 / C3 / DyMoS / packcache_decay / "
                                 "fp_pool / forget / UnStep / guidance / cadence / spectral / Phase-1 ports")
            if not self._relativistic_rope:
                raise ValueError(f"factorial.{name} needs model_kwargs.relativistic_rope")
            if getattr(sa, "rope_theta_by_head", None) is not None:
                raise ValueError(f"factorial.{name} is not defined with per-head RoPE bases")
            if self.echo_enabled:
                if name == "tether_cache":
                    raise ValueError("factorial.tether_cache replaces the sink + FIFO window: defined on the static sink only")
                if getattr(self, "echo_pin_latent", None) != 0:
                    raise ValueError(f"factorial.{name} on Echo needs factorial.echo_pin_source (the source is the sink)")
            elif int(sa.sink_size) < 1 or self.pure_sliding_window or bool(sa.ema_sink):
                raise ValueError(f"factorial.{name}: needs the static sink (sink_size >= 1, ema_sink false, no pure "
                                 "sliding window) or Echo + pinsrc")
            fs = int(sa.frame_seq_length)
            W = int(sa.max_attention_size // fs)
            blocks = self.generator.model.blocks
            if name == "forcing_kv":
                prof = pcfg.get("head_profile", "configs/factorial/forcingkv_heads_causalforcing.json")
                prof_sha = None
                import hashlib, json, os
                if isinstance(prof, str):
                    path = prof if os.path.isabs(prof) else os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), prof)
                    raw = open(path, "rb").read()
                    prof_sha = hashlib.sha256(raw).hexdigest()
                    prof = json.loads(raw)
                prof = dict(prof)
                nh = int(blocks[0].self_attn.num_heads)
                layers = {int(it.get("layer_idx", i)): it for i, it in enumerate(prof.get("layers", []))}
                if sorted(layers) != list(range(len(blocks))):
                    raise ValueError(f"factorial.forcing_kv.head_profile covers layers {sorted(layers)}, the model has "
                                     f"{len(blocks)}")
                seg, ret = int(pcfg.get("segments", 6)), float(pcfg.get("retention", 0.33))
                cand, score_layer = int(pcfg.get("candidates", 3)), int(pcfg.get("score_layer", 1))
                if seg < 1 or not 0.0 <= ret <= 1.0 or cand < 1 or not 1 <= score_layer < len(blocks):
                    raise ValueError("factorial.forcing_kv: segments >= 1, retention in [0, 1], candidates >= 1, "
                                     "score_layer in [1, layers)")
                n_dyn_pairs = 0
                groups = []
                for i, block in enumerate(blocks):
                    st_h = [int(h) for h in layers[i].get("static_head", [])]
                    dy_h = [int(h) for h in layers[i].get("dynamic_head", [])]
                    if sorted(st_h + dy_h) != list(range(nh)):
                        raise ValueError(f"factorial.forcing_kv.head_profile layer {i}: heads must cover 0..{nh - 1} once")
                    if i == score_layer and not dy_h:
                        raise ValueError("factorial.forcing_kv: the score layer has no dynamic head")
                    if i > 0:
                        n_dyn_pairs += len(dy_h)
                    groups.append((st_h, dy_h))
                shared = {}
                frac = n_dyn_pairs / float(len(blocks) * nh)
                for i, block in enumerate(blocks):
                    st_h, dy_h = groups[i]
                    block.self_attn.forcing_kv = dict(
                        layer=i, score_layer=score_layer, segments=seg, retention=ret, candidates=cand, shared=shared,
                        static_heads=torch.tensor(st_h, dtype=torch.long) if i > 0 else torch.arange(nh),
                        dynamic_heads=torch.tensor(dy_h, dtype=torch.long) if i > 0 else None, dynamic_fraction=frac)
                    block.self_attn.pack_stats = []
                self.forcing_kv = dict(head_profile=pcfg.get("head_profile", "configs/factorial/forcingkv_heads_causalforcing.json")
                                       if isinstance(pcfg.get("head_profile", "x"), str) else "inline",
                                       head_profile_sha256=prof_sha, profile_label=prof.get("profile"), segments=seg,
                                       retention=ret, candidates=cand, score_layer=score_layer,
                                       dynamic_head_pairs=n_dyn_pairs, dynamic_fraction=frac,
                                       buffer_tokens=int(round(cand * seg * ret)) * fs // seg)
                print(f"[factorial] forcing_kv (Forcing-KV arXiv 2605.09681, released-code behaviour): head profile "
                      f"{self.forcing_kv['head_profile']} ({prof.get('profile')}, sha256 {prof_sha}); static heads attend "
                      f"[source, newest history latent, current block], dynamic heads (layers >= 1: {n_dyn_pairs} of "
                      f"{len(blocks) * nh} head-layers) + the {int(round(cand * seg * ret))} least adjacent-similar of "
                      f"{cand} x {seg} bands of the {cand} latents before it (score layer {score_layer}, shared); layer 0 "
                      f"static; read-time only")
            elif name == "future_kv":
                budget = str(pcfg.get("budget", "uni"))
                if budget not in ("uni", "rho03"):
                    raise ValueError("factorial.future_kv.budget must be uni (fg_uni) | rho03 (fg_rho03)")
                rho, rec_s = (0.5, 0.75) if budget == "uni" else (0.3, 0.0)
                lam, hor = float(pcfg.get("lam", 0.5)), int(pcfg.get("horizon", 6))
                tau, w, merge = float(pcfg.get("tau", 0.95)), int(pcfg.get("w", 3)), bool(pcfg.get("merge", True))
                if not 0.0 <= lam <= 1.0 or hor < 1 or w < 1 or not -1.0 <= tau <= 1.0:
                    raise ValueError("factorial.future_kv: lam in [0, 1], horizon >= 1, w >= 1, tau in [-1, 1]")
                self.future_kv = dict(budget=budget, rho=rho, recent_s=rec_s, recent_latents=int(round(rec_s / 0.25)),
                                      lam=lam, horizon=hor, tau=tau, w=w, merge=merge)
                for block in blocks:
                    block.self_attn.future_kv = dict(self.future_kv)
                    block.self_attn._ff_state = None
                    block.self_attn.pack_stats = []
                print(f"[factorial] future_kv (Future Forcing arXiv 2605.30083, our port): per layer, once per block, keep "
                      f"the top-B alive history tokens by lambda {lam} x future-proxy attention (mean pre-RoPE query of the "
                      f"last {w} committed latents, RoPE at +1..+{hor}) + (1 - lambda) x average historical attention; B = "
                      f"the fg_{budget} budget (rho {rho}, recent {rec_s} s); evicted tokens "
                      f"{'merged into the most profile-similar retained token (cos >= ' + str(tau) + ', Eq. 3 values)' if merge else 'dropped'}; "
                      f"persistent port state (host cache never written); source and current block whole")
            else:
                mem, rec = int(pcfg.get("memory", 14)), int(pcfg.get("recent", 4))
                alpha, tau, temp = float(pcfg.get("alpha", 0.35)), float(pcfg.get("tau", 0.6)), float(pcfg.get("temperature", 1.0))
                if int(sa.sink_size) + mem + rec != W:
                    raise ValueError(f"factorial.tether_cache: sink {int(sa.sink_size)} + memory {mem} + recent {rec} must "
                                     f"equal the window {W} (the paper's K = 21 = 3 + 14 + 4)")
                if not 1 <= nfpb <= rec or not 0.0 <= tau <= 1.0 or alpha < 0 or temp <= 0:
                    raise ValueError("factorial.tether_cache: 1 <= num_frame_per_block <= recent, tau in [0, 1], "
                                     "alpha >= 0, temperature > 0")
                self.tether_cache = dict(memory=mem, recent=rec, alpha=alpha, tau=tau, temperature=temp,
                                         sink=int(sa.sink_size), window=W)
                for block in blocks:
                    block.self_attn.tether_cache = dict(self.tether_cache)
                    block.self_attn._tc_state = None
                    block.self_attn.pack_stats = []
                print(f"[factorial] tether_cache (TetherCache arXiv 2606.13035, released-code behaviour): window = sink "
                      f"{int(sa.sink_size)} | memory {mem} (per-layer store, GRAB alpha {alpha}, TAME tau {tau}) | recent "
                      f"{rec} incl. the current block; warm-up = the host path; memory written in the port's own store")
        # ---- UnStep port (arXiv 2609.32518; tools/baselines/unstep for the Self-Forcing port) ----
        # local_layer_sizes  per-layer READ window "a-b:n;c:m" (later entries override): layer i attends the LAST
        #                    n frames of its KV buffer (model_kwargs.local_attn_size), exactly as UnStep's wrapper sets
        #                    per-layer max_attention_size; the buffer, eviction and the dual sink are unchanged. A read
        #                    window shorter than the buffer does not contain slot 0 (UnStep: layers reading 8 of a
        #                    9-frame buffer see sink slots 1-2 + 6 recent, not slot 0): those keys are all re-roped.
        # clean_sigma        C5: the clean refresh re-noises the block's x0 at sigma* (x_s = (1-s) x0 + s eps), keeps
        #                    the t = 0 label, writes the KV cache from x_s, and EMITS x0' = x_s - s * flow.
        # svd_vo             C6: "a-b:ratio;..." truncated SVD of each self-attention V and O projection at load
        #                    (rank = round(min(in, out) * ratio); 1.0 = untouched), applied by inference.py through
        #                    apply_svd_vo() after the checkpoint is loaded; ranks recorded in RUN.json.
        lls = fac.get("local_layer_sizes", None)
        self.local_layer_sizes = None
        if lls:
            if self.echo_enabled:
                raise ValueError("factorial.local_layer_sizes is not defined on Echo (its A / C / R budget is the window)")
            if not self._relativistic_rope or int(self.local_attn_size) == -1:
                raise ValueError("factorial.local_layer_sizes needs relativistic_rope and a bounded local_attn_size buffer")
            if self.pure_sliding_window or self.source_schedule is not None or self.role_routing is not None \
                    or self.dymos_ref_scale is not None or self.packcache_decay is not None or self.fp_pool is not None \
                    or self.forget is not None:
                raise ValueError("factorial.local_layer_sizes is a single-mechanism port: the C2 / C3 / DyMoS / PackCache / "
                                 "fp_pool arms assume the window starts at the source slot")
            sizes = self._parse_layer_spec(str(lls), self.num_transformer_blocks, int)
            buf = int(self.local_attn_size)
            if int(sa.sink_size) + nfpb > buf:
                raise ValueError(f"factorial.local_layer_sizes: buffer {buf} < sink {int(sa.sink_size)} + block {nfpb}")
            for i, n in sizes.items():
                if not nfpb <= n <= buf:
                    raise ValueError(f"factorial.local_layer_sizes: layer {i} reads {n} frames, must be in "
                                     f"[num_frame_per_block {nfpb}, buffer {buf}]")
            per_layer = []
            for i, block in enumerate(self.generator.model.blocks):
                n = sizes.get(i, buf)
                block.self_attn.max_attention_size = n * int(block.self_attn.frame_seq_length)
                per_layer.append(n)
            self.local_layer_sizes = dict(spec=str(lls), buffer=buf, per_layer_read=per_layer)
            print(f"[factorial] local_layer_sizes '{lls}': KV buffer {buf} frames (sink {int(sa.sink_size)}"
                  f"{' dual EMA' if sa.ema_sink else ''}); per-layer read frames {per_layer}")
        cs = fac.get("clean_sigma", None)
        self.clean_sigma = None
        if cs is not None:
            cs = float(cs)
            if not 0.0 < cs < 1.0:
                raise ValueError("factorial.clean_sigma (UnStep C5 sigma*) must be in (0, 1)")
            if float(getattr(args, "context_noise", 0) or 0) != 0:
                raise ValueError("factorial.clean_sigma replaces the t = 0 refresh: context_noise must be 0")
            if self.role_routing is not None:
                raise ValueError("factorial.clean_sigma is not defined with role_routing")
            self.clean_sigma = cs
            print(f"[factorial] clean_sigma={cs}: UnStep C5 clean-cache refinement: the refresh pass sees "
                  f"x_s = (1-{cs}) x0 + {cs} eps at label t=0, writes the cache from it and emits x0' = x_s - {cs} * flow")
        sv = fac.get("svd_vo", None)
        self.svd_vo = None
        if sv:
            ratios = self._parse_layer_spec(str(sv), self.num_transformer_blocks, float)
            if any(not 0.0 < r <= 1.0 for r in ratios.values()):
                raise ValueError("factorial.svd_vo: keep ratios must be in (0, 1]")
            self.svd_vo = dict(spec=str(sv), ratios=[ratios.get(i, 1.0) for i in range(self.num_transformer_blocks)],
                               applied=False, ranks=None)
            print(f"[factorial] svd_vo '{sv}': UnStep C6 truncated SVD of self-attention V and O, applied at load")
        # ---- / "student qualification": PROMPT EXTRAPOLATION (not CFG) ----------------------
        # guidance {mode: extrapolate, scale: s, branch: empty|neg_official|neg_nostatic|counter|same,
        #           counter_prompts: <json>}: at every denoising evaluation a second forward of the frozen student with
        # the alternative text on the SAME noisy input, image prefix and committed past; the clean prediction becomes
        # z = z_base + (s - 1)(z_base - z_alt) (equivalently on the flow: x0 = x_t - sigma * flow is affine in flow at
        # fixed x_t, sigma). The student is DMD-distilled with guidance baked in, so this extrapolates an already-guided
        # generator; it does not restore a missing CFG score. The alternative branch is READ-ONLY on the persistent
        # state (its writes into the current-block KV slots and the end indices are restored; Echo commits only at
        # the refresh), consumes no RNG, uses its own text cross-attention cache, and the clean refresh is one pass
        # of the base branch on z. s = 1 skips the branch entirely (bitwise the default).
        gd = fac.get("guidance", None)
        self.guidance = None
        if gd:
            mode = str(gd.get("mode", "extrapolate"))
            if mode != "extrapolate":
                raise ValueError("factorial.guidance.mode must be 'extrapolate' (prompt extrapolation of the distilled student)")
            scale = float(gd.get("scale", 1.0))
            if not 0.0 < scale <= 10.0:
                raise ValueError("factorial.guidance.scale must be in (0, 10]")
            branch = str(gd.get("branch", "empty"))
            if branch not in ("empty", "neg_official", "neg_nostatic", "counter", "same"):
                raise ValueError("factorial.guidance.branch must be empty | neg_official | neg_nostatic | counter | same")
            others = dict(source_schedule=self.source_schedule, role_routing=self.role_routing,
                          dymos_ref_scale=self.dymos_ref_scale, packcache_decay=self.packcache_decay,
                          fp_pool=self.fp_pool, forget=self.forget, local_layer_sizes=self.local_layer_sizes,
                          clean_sigma=self.clean_sigma,
                          echo_cadence_latents=self.echo_cadence_latents)
            on = [k for k, v in others.items() if v is not None]
            if on or self.alg_enabled or float((p1.get("dymos", {}) or {}).get("gamma", 0.0)) \
                    or float(getattr(sa, "echo_spectral_strength", 0.0)) > 0 or getattr(sa, "material_reader", None) is not None:
                raise ValueError(f"factorial.guidance is a single-mechanism qualification arm: not with {on or 'the Phase-1 / spectral / reader paths'}")
            text, counter = None, None
            if branch == "empty":
                text = ""
            elif branch in ("neg_official", "neg_nostatic"):
                text = _negative_controls().negative_condition(
                    WAN22_SAMPLE_NEG_PROMPT, "official" if branch == "neg_official" else "without_static")["text"]
            elif branch == "counter":
                path = gd.get("counter_prompts", None)
                if not path:
                    raise ValueError("factorial.guidance.branch=counter needs counter_prompts: <json>")
                import hashlib as _hashlib
                import json as _json
                raw = open(path, "rb").read()          # relative to the working directory (the repo root)
                counter = dict(path=str(path), sha256=_hashlib.sha256(raw).hexdigest(),
                               by_file={p["file_name"]: p for p in _json.loads(raw.decode("utf-8"))["pairs"]})
            self.guidance = dict(mode=mode, scale=scale, branch=branch, text=text, counter=counter)
            self.guidance_alt_text = None      # set per video by inference.py for branch counter
            if scale != 1.0:
                for block in self.generator.model.blocks:
                    block.self_attn.guidance_active = True     # the base pass records its query RoPE position
            print(f"[factorial] guidance: PROMPT EXTRAPOLATION (not CFG: the DMD student has guidance baked in) "
                  f"z = z_base + ({scale} - 1)(z_base - z_alt), branch {branch}"
                  + (f" ({len(text)} chars)" if text is not None else "")
                  + (f" ({counter['path']}, sha256 {counter['sha256'][:12]})" if counter else "")
                  + f"; {'2x DiT forwards on every denoising evaluation, refresh single pass' if scale != 1.0 else 's = 1: branch skipped'}")
        if self.refine_last is not None and (self.source_schedule is not None or self.role_routing is not None
                                             or self.dymos_ref_scale is not None or self.guidance is not None
                                             or self.alg_enabled):
            raise ValueError("factorial.refine_last is a sampler control: not with C2 / C3 / DyMoS / guidance / ALG "
                             "(their per-evaluation tables assume a strictly decreasing step list)")
        self.compose = bool(fac.get("compose", False))
        # fixed-camera locality prior (opt-in): on the packed window, each current-block query gets the
        # additive bias -((max(0, dist - radius)) / sigma)^2 to every source / history key at grid distance dist (token
        # units, 16 px); keys of the current block are unbiased (within-block attention stays global)
        lc = fac.get("locality", None)
        self.locality = None
        if lc:
            lc = lc if hasattr(lc, "get") else {}                   # mapping / DictConfig; `true` = defaults
            radius, sigma = float(lc.get("radius", 4.0)), float(lc.get("sigma", 4.0))
            if radius < 0 or sigma <= 0:
                raise ValueError("factorial.locality: radius >= 0, sigma > 0")
            if self.packcache_decay is None and self.forget is None and self.fp_pool is None:
                raise ValueError("factorial.locality acts on the packed window: needs packcache_decay / forget / fp_pool")
            if self.compose or self.source_schedule is not None or self.dymos_ref_scale is not None:
                raise ValueError("factorial.locality is not composed with C2 / DyMoS (one biased kernel per read)")
            self.locality = dict(radius=radius, sigma=sigma)
            for block in self.generator.model.blocks:
                block.self_attn.locality = dict(self.locality)
            print(f"[factorial] locality prior: bias -((max(0, d - {radius})) / {sigma})^2 from every current-block query to "
                  f"each source / history key at grid distance d (tokens); current-block keys unbiased")
        if self.compose:
            on = [n for n, v in (("packcache_decay", self.packcache_decay), ("forget", self.forget),
                                 ("source_schedule", self.source_schedule), ("dymos_ref_scale", self.dymos_ref_scale)) if v is not None]
            print(f"[factorial] compose: {' + '.join(on) or 'nothing to compose'}: the history budget thins "
                  f"generated frames only; C2 / DyMoS act on the whole source entry inside the packed window)")
        # C1 budget accounting (logging only; no tensor the output depends on is touched):
        # per block, layer 0 records the attended key frames, their latent ids (duplicates, coverage of
        # the last 21 latents, source attended) and the subsampled source-key attention mass; the
        # pipeline adds the static KV allocation. inference.py writes <video>_context.json + RUN.json.
        self.context_accounting = bool(fac.get("context_accounting", False))
        self.context_rows = []
        if fac:
            print(f"[factorial] pure_sliding_window={self.pure_sliding_window} "
                  f"spectral_anchor_frames={spectral_frames} "
                  f"warm_start_frames={self.warm_start_frames} "
                  f"latent_indexed_renoise={self.latent_indexed_renoise} "
                  f"warm_start_frames_nosplit={self.warm_start_frames_nosplit} "
                  f"echo_cadence_latents={self.echo_cadence_latents} "
                  f"ema_alpha_normalize={self.ema_alpha_normalize} "
                  f"source_schedule={self.source_schedule} role_routing={self.role_routing} "
                  f"dymos_ref_scale={self.dymos_ref_scale} context_accounting={self.context_accounting}")

    @staticmethod
    def _parse_layer_spec(spec, n_layers, cast=int):
        """'8-29:8;8:9' -> {layer: value}; later entries override earlier ones (unstep.wrapper_runtime /
        tools/baselines/unstep/run_unstep_sf_i2v.py parse_layer_spec)."""
        out = {}
        for item in str(spec).split(";"):
            item = item.strip()
            if not item:
                continue
            span, val = item.split(":", 1)
            a, b = ((int(x) for x in span.split("-", 1)) if "-" in span else (int(span), int(span)))
            if a < 0 or b < a or b >= n_layers:
                raise ValueError(f"layer span {span!r} outside 0..{n_layers - 1}")
            for layer in range(a, b + 1):
                out[layer] = cast(val)
        return out

    def apply_svd_vo(self):
        """UnStep C6: replace each self-attention V and O projection with its truncated-SVD
        reconstruction (computed in float32 from the loaded weights, materialised as a dense nn.Linear in the
        weight's dtype, bias kept; rank = round(min(in, out) * ratio), ratio 1.0 = untouched), exactly as
        tools/baselines/unstep/run_unstep_sf_i2v.py truncated_svd_linear. Call after the checkpoint is loaded.
        Returns per-layer rows {block, keep_ratio, v_rank, o_rank, full_rank}."""
        if self.svd_vo is None:
            return []
        if self.svd_vo["applied"]:
            raise RuntimeError("svd_vo already applied")
        rows = []
        for i, block in enumerate(self.generator.model.blocks):
            ratio = float(self.svd_vo["ratios"][i])
            row = dict(block=i, keep_ratio=ratio)
            for t in ("v", "o"):
                lin = getattr(block.self_attn, t)
                full = min(lin.in_features, lin.out_features)
                r = min(max(1, int(round(full * ratio))), full)
                row[f"{t}_rank"], row["full_rank"] = r, full
                if r < full:
                    dense = torch.nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                            device=lin.weight.device, dtype=lin.weight.dtype)
                    with torch.no_grad():
                        u, s, vh = torch.linalg.svd(lin.weight.float(), full_matrices=False)
                        dense.weight.copy_(((u[:, :r] * s[:r].unsqueeze(0)) @ vh[:r, :]).to(lin.weight.dtype))
                        if lin.bias is not None:
                            dense.bias.copy_(lin.bias)
                    setattr(block.self_attn, t, dense.requires_grad_(False))
            rows.append(row)
        self.svd_vo["applied"] = True
        self.svd_vo["ranks"] = rows
        n_cut = sum(1 for r in rows if r["v_rank"] < r["full_rank"])
        print(f"[factorial] svd_vo applied: {n_cut}/{len(rows)} layers truncated; ranks "
              + ", ".join(f"{r['block']}:{r['v_rank']}/{r['full_rank']}" for r in rows if r['v_rank'] < r['full_rank'])[:400])
        return rows

    # ---- prompt extrapolation -------------------------------------------------------------------
    def _guidance_setup(self, text_prompts, conditional_dict, batch_size, noise):
        """Per video: the alternative text conditioning, its own (empty) text cross-attention cache and the record.
        None when guidance is off or s == 1 (the branch is skipped entirely: bitwise the default)."""
        g = getattr(self, "guidance", None)
        if g is None or g["scale"] == 1.0:
            self.guidance_record = None if g is None else dict(branch=g["branch"], scale=g["scale"], skipped=True)
            return None
        if g["branch"] == "same":
            alt = text_prompts[0]
        elif g["branch"] == "counter":
            alt = getattr(self, "guidance_alt_text", None)
            if alt is None:
                raise ValueError("guidance branch counter: pipeline.guidance_alt_text not set for this video "
                                 "(inference.py looks it up by the image file name)")
        else:
            alt = g["text"]
        alt_dict = self.text_encoder(text_prompts=[alt])
        for key in ("clip_fea",):
            if key in conditional_dict:
                alt_dict[key] = conditional_dict[key]
        live = self.crossattn_cache                          # a separate, freshly initialised text cache
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self.crossattn_cache_alt, self.crossattn_cache = self.crossattn_cache, live
        self.guidance_record = dict(branch=g["branch"], scale=g["scale"], base_text=text_prompts[0], alt_text=alt,
                                    evals=[], cost=dict(base_denoise_forwards=0, alt_forwards=0, refresh_forwards=0),
                                    note="prompt extrapolation of a DMD-distilled student (guidance baked in), not CFG")
        return alt_dict

    def _guidance_extrapolate(self, z_base, noisy_input, alt_dict, conditional_dict, timestep, current_start_frame,
                              index, schedule):
        """z = z_base + (s - 1)(z_base - z_alt), z_alt from the frozen student with the alternative text on the SAME
        noisy input and committed past. Read-only: the base pass of this evaluation has already written the
        current block's noisy K/V and done any eviction (asserted: global end == block end on every layer); the
        alternative pass writes only the current-block slots and the end indices, which are saved and restored, and
        Echo commits only at the clean refresh; context / pack logging is suppressed. No RNG is drawn."""
        from wan21.modules import causal_model as _cm
        g = self.guidance
        fs = self.frame_seq_length
        n_new = noisy_input.shape[1] * fs
        block_end = current_start_frame * fs + n_new
        saved = []
        for c in self.kv_cache1:
            ge, le = int(c["global_end_index"]), int(c["local_end_index"])
            if ge != block_end:
                raise RuntimeError(f"guidance: the base pass must run first (global end {ge} != block end {block_end})")
            if self.echo_enabled:
                saved.append((None, None, ge, le))
            else:
                saved.append((c["k"][:, le - n_new:le].clone(), c["v"][:, le - n_new:le].clone(), ge, le))
        if "y" in conditional_dict:
            alt_dict["y"] = conditional_dict["y"]
        _cm.set_branch_readonly(True)
        try:
            _, z_alt = self.generator(noisy_image_or_video=noisy_input, conditional_dict=alt_dict, timestep=timestep,
                                      kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache_alt,
                                      current_start=current_start_frame * fs)
        finally:
            _cm.set_branch_readonly(False)
            for c, (k, v, ge, le) in zip(self.kv_cache1, saved):
                if k is not None:
                    c["k"][:, le - n_new:le].copy_(k)
                    c["v"][:, le - n_new:le].copy_(v)
                c["global_end_index"].fill_(ge)
                c["local_end_index"].fill_(le)
        s = float(g["scale"])
        ct = torch.promote_types(z_base.dtype, torch.float32)       # bf16 -> fp32; fp32 / fp64 unchanged
        diff = z_base.to(ct) - z_alt.to(ct)
        z = (z_base.to(ct) + (s - 1.0) * diff).to(z_base.dtype)    # same text: diff == 0 -> z == z_base bitwise
        rec = self.guidance_record
        rec["cost"]["base_denoise_forwards"] += 1
        rec["cost"]["alt_forwards"] += 1
        if index == 0:
            rec["cost"]["refresh_forwards"] += 1                  # one clean refresh per block, single pass
        rec["evals"].append(dict(block_start=int(current_start_frame), index=int(index),
                                 t_model=round(float(schedule[index]), 3),
                                 rms_diff=float(diff.pow(2).mean().sqrt()), rms_base=float(z_base.float().pow(2).mean().sqrt())))
        return z

    def _sigma_of(self, t_model):
        """Scheduler sigma at the (warped) timestep the model receives (nearest entry, like add_noise)."""
        ts = self.scheduler.timesteps.detach().float().cpu()
        i = int((ts - float(t_model)).abs().argmin())
        return float(self.scheduler.sigmas.detach().float().cpu()[i])

    def _context_static(self):
        """Static KV accounting per video: the allocated cache (every layer's k + v tensors, bytes), the
        bytes of one attended key frame summed over layers (K + V), and the two C2 source entries
        (counted separately: they live outside the cache / Echo memory)."""
        fs = self.frame_seq_length
        layers = self.num_transformer_blocks
        alloc = sum(int(c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size())
                    for c in self.kv_cache1)
        c0 = self.kv_cache1[0]["k"]
        per_frame = int(fs * c0.shape[2] * c0.shape[3] * c0.element_size() * 2 * layers)
        sa = self.generator.model.blocks[0].self_attn
        return dict(kv_cache_allocated_bytes=alloc, kv_cache_allocated_frames=int(c0.shape[1] // fs),
                    bytes_per_attended_frame_all_layers=per_frame, layers=layers, frame_seq_length=fs,
                    window_frames=int(sa.max_attention_size // fs), nfpb=int(self.num_frame_per_block),
                    policy=("echo" if self.echo_enabled else f"sink{int(sa.sink_size)}" + ("_ema" if sa.ema_sink else "")),
                    c2_source_entries=2 if self.source_schedule else 1,
                    c2_source_entries_bytes=(2 * per_frame if self.source_schedule else 0),
                    echo_pin_latent=getattr(self, "echo_pin_latent", None))

    # ---- C2: two-entry source schedule ------------------------------------------------
    def _source_reads_soft(self, nominal_list, index):
        """Entry read by evaluation `index` of a block whose NOMINAL schedule is `nominal_list`
        (config units, e.g. [1000, 500]; the model sees the warped values). by_noise: soft iff the
        nominal timestep >= t_switch (default: smallest regular timestep + 1, so the regular
        schedule reads soft at 1000 and sharp at 500, and the first block reads soft at 1000 / 750
        and sharp at 500 / 250 -- one noise threshold for both schedules). last_sharp: every
        evaluation but the last of its own schedule reads soft (first block 1000 / 750 / 500 soft,
        250 sharp) -- indexed by step position, kept as an opt-in. always_soft: every denoising
        evaluation reads soft (the refresh and every commit stay sharp): the "always soft" control
        (launcher arm c2soft; the default cell is the all-sharp control, "c2sharp")."""
        ss = self.source_schedule
        if ss["steps"] == "always_soft":
            return True
        if ss["steps"] == "last_sharp":
            return index < len(nominal_list) - 1
        return int(nominal_list[index]) >= ss["t_switch"]

    def _attn_modules(self):
        return [block.self_attn for block in self.generator.model.blocks]

    def _capture_source_pass(self, latent, conditional_dict):
        """K/V entry per layer that the unchanged source pass (Step 2 of inference) would STORE
        for `latent`, computed on throw-away state: a scratch copy of every layer's KV cache
        (indices reset to 0 so the pass writes slot 0) and a snapshot / restore of the Echo
        memory (whose commit of tick 0 would otherwise replace the real entry 0 and move the
        query centre). The cross-attention cache is already initialised and read-only here."""
        fs = self.frame_seq_length
        attns = self._attn_modules()
        echo_state = [a._echo_state_copy() for a in attns] if self.echo_enabled else None
        # one-frame scratch caches (indices 0): the pass writes slot 0 without eviction and ropes a
        # 1-frame grid, exactly like the real source pass into the empty full cache; 1/21 the memory
        scratch = [{"k": torch.zeros_like(c["k"][:, :fs]), "v": torch.zeros_like(c["v"][:, :fs]),
                    "global_end_index": torch.zeros_like(c["global_end_index"]),
                    "local_end_index": torch.zeros_like(c["local_end_index"])}
                   for c in self.kv_cache1]
        timestep = torch.zeros([latent.shape[0], 1], device=latent.device, dtype=torch.int64)
        try:
            self.generator(noisy_image_or_video=latent, conditional_dict=conditional_dict,
                           timestep=timestep, kv_cache=scratch, crossattn_cache=self.crossattn_cache,
                           current_start=0)
            if self.echo_enabled:
                out = [(a._echo_early_k[0].clone(), a._echo_early_v[0].clone()) for a in attns]
            else:
                out = [(c["k"][:, :fs].clone(), c["v"][:, :fs].clone()) for c in scratch]
        finally:
            if echo_state is not None:
                for a, st in zip(attns, echo_state):
                    for name, val in st.items():
                        setattr(a, name, val)
        return out

    def _prepare_xref_reference(self, conditional_dict):
        """ fg_srcred_xref: the per-layer V entry that the unchanged source pass would store for ANOTHER
        dev scene's image latent (pipeline.forget_xref_latent), computed by `_capture_source_pass` on a one-frame scratch
        cache with the Echo memory snapshot / restored - nothing is written to the real cache - under THIS scene's text
        conditioning (only the image differs). Installed as each layer's redundancy reference."""
        lat = self.forget_xref_latent
        entries = self._capture_source_pass(lat.to(device=self.kv_cache1[0]["k"].device, dtype=self.kv_cache1[0]["k"].dtype),
                                            conditional_dict)
        for a_, (k_, v_) in zip(self._attn_modules(), entries):
            a_.forget_ref_v = v_
        self.forget_xref_record = dict(scene=getattr(self, "forget_xref_scene", None), layers=len(entries))

    def _prepare_source_entries(self, source_latent, conditional_dict):
        """Build the sharp and the soft source entries once per video, right after the real
        source pass. sharp = the entry the default path just stored (read from the live cache /
        Echo entry 0, so bitwise the default's). soft = lowpass: the stored entry of the Gaussian
        low-passed source latent (sigma in latent px, _latent_lowpass); downweight: (sharp K,
        weight * sharp V). Nothing here writes to the cache or the Echo memory."""
        ss = self.source_schedule
        fs = self.frame_seq_length
        attns = self._attn_modules()
        if self.echo_enabled:
            for a in attns:
                if a._echo_early_ids[:1] != [0]:
                    raise RuntimeError("source_schedule: Echo entry 0 is not the source latent")
            sharp = [(a._echo_early_k[0], a._echo_early_v[0]) for a in attns]
        else:
            sharp = [(c["k"][:, :fs].clone(), c["v"][:, :fs].clone()) for c in self.kv_cache1]
        if ss["mode"] == "downweight":
            w = ss["weight"]
            soft = [(k, v * w) for k, v in sharp]
        else:
            soft = self._capture_source_pass(self._latent_lowpass(source_latent, ss["sigma"]),
                                             conditional_dict)
        for a, sh, so in zip(attns, sharp, soft):
            a.source_entries = {"sharp": sh, "soft": so}
            a.source_read = "sharp"
            a.source_role = None
        self._source_logged = set()

    def _set_source_step(self, block_index, warm_block, index, n_steps):
        """Per evaluation: which source entry attention reads (C2) and whether DyMoS is active.
        `index` None = the clean refresh (t = 0): sharp, DyMoS off."""
        if self.source_schedule is not None:
            if index is None:
                read = "sharp"
            else:
                name = "first" if warm_block and self.denoising_step_list_first_chunk is not None else "regular"
                lst = self._nominal_first if name == "first" else self._nominal_regular
                read = "soft" if self._source_reads_soft(lst, index) else "sharp"
                if name not in self._source_logged:
                    self._source_logged.add(name)
                    print(f"[source_schedule] block {block_index} ({name} schedule): "
                          + " ".join(f"{t}->{r}" for t, r in self._source_table[name])
                          + "; refresh->sharp")
                    print(f"[source_schedule][model] block {block_index} ({name} schedule): "
                          + " ".join(f"nominal {r['t_nominal']} -> model t {r['t_model']:.0f} (sigma {r['sigma']:.3f}, "
                                     f"alpha {r['alpha']:.3f}) {r['entry']}"
                                     for r in self.schedule_record[name] if r["index"] is not None)
                          + "; refresh t 0 (sigma 0) sharp")
            for a in self._attn_modules():
                a.source_read = read
        if self.dymos_ref_scale is not None:
            active = index is not None and index < self.dymos_ref_scale["early_steps"]
            for a in self._attn_modules():
                a.dymos_active = active

    # ---- C3: online, model-intrinsic role map -----------------------------------------
    @staticmethod
    def _assert_no_mask_config(args):
        """C3 must never see an evaluation mask (results/factorial/source_masks, SNF masks, region /
        anchor masks): refuse any configured key whose name contains 'mask' with a truthy value."""
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(args, resolve=True) if not isinstance(args, dict) else args
        except Exception:
            cfg = args if isinstance(args, dict) else {}

        def walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    if "mask" in str(k).lower() and v:
                        raise ValueError(f"factorial.role_routing is model-intrinsic: refuse the "
                                         f"configured mask key '{path}{k}' = {v!r}")
                    walk(v, f"{path}{k}.")
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    walk(v, f"{path}{i}.")
        walk(cfg, "")

    @staticmethod
    def _otsu_threshold(x):
        """Otsu's threshold of a 1-D tensor (256-bin histogram, maximal between-class variance)."""
        x = x.float()
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return hi
        bins = 256
        hist = torch.histc(x, bins=bins, min=lo, max=hi)
        centres = lo + (torch.arange(bins, dtype=torch.float32, device=x.device) + 0.5) * (hi - lo) / bins
        w0 = hist.cumsum(0)
        w1 = w0[-1] - w0
        m0 = (hist * centres).cumsum(0) / w0.clamp_min(1e-12)
        m1 = ((hist * centres).sum() - (hist * centres).cumsum(0)) / w1.clamp_min(1e-12)
        between = w0 * w1 * (m0 - m1) ** 2
        between[w0 == 0] = -1
        between[w1 == 0] = -1
        best = (between >= between.max() - 1e-9 * between.max().abs()).nonzero(as_tuple=True)[0]
        i = int(best[len(best) // 2])          # midpoint of a flat optimum (empty bins between two modes)
        return lo + (i + 1) * (hi - lo) / bins

    # ---- material-aware forgetting: online role map refreshed in video time ---------------------
    def _volatility_roles(self, frames, threshold):
        """The C3 estimator on a window of the student's own clean latents [n, C, H, W]: per-pixel temporal variance
        (population), mean over channels, mean-pooled over the DiT patch to the attention token grid, log10, Otsu or a
        quantile; returns (log10 variance grid, threshold, material bool flat [tokens])."""
        x = frames.float()
        var = (x.pow(2).mean(0) - x.mean(0).pow(2)).clamp_min(0).mean(0)
        ps = getattr(self.generator.model, "patch_size", (1, 2, 2))
        tok = torch.nn.functional.avg_pool2d(var[None, None], kernel_size=(int(ps[1]), int(ps[2])))[0, 0]
        log_var = torch.log10(tok + 1e-12)
        flat = log_var.flatten()
        if flat.numel() != self.frame_seq_length:
            raise RuntimeError(f"forget map has {flat.numel()} tokens, attention has {self.frame_seq_length}")
        thr = self._otsu_threshold(flat) if threshold == "auto" else float(torch.quantile(flat, float(threshold)))
        return log_var, float(thr), flat > thr

    def _forget_reset(self):
        f = self.forget
        self._forget_buf, self._forget_n, self._forget_j = [], 0, 0
        self._prev_dense_frozen = False                 #: the first map's dense mask not installed yet
        self.forget_events = []
        self._forget_perm = None
        for a_ in self._attn_modules():
            a_.forget_ref_v = None
            a_.forget_ref_perm = None
        if f.get("reference") == "pshuf":
            #: one fixed token-position permutation of the source V per video (seeded by the video seed)
            seed = getattr(self, "renoise_seed", None)
            if seed is None:
                raise RuntimeError("factorial.forget.reference=pshuf needs the video seed (pipeline.renoise_seed)")
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 23)
            self.forget_ref_perm = torch.randperm(self.frame_seq_length, generator=g)
            for a_ in self._attn_modules():
                a_.forget_ref_perm = self.forget_ref_perm
        if f.get("reference") == "xref" and getattr(self, "forget_xref_latent", None) is None:
            raise RuntimeError("factorial.forget.reference=xref needs pipeline.forget_xref_latent (inference.py encodes "
                               "the deranged dev scene's image)")
        if f["shuffle"]:
            seed = getattr(self, "renoise_seed", None)
            if seed is None:
                raise RuntimeError("factorial.forget.shuffle needs the video seed (pipeline.renoise_seed)")
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 11)
            self._forget_perm = torch.randperm(self.frame_seq_length, generator=g)
        for a in self._attn_modules():
            a.forget_role = None
            if any(k in f for k in ("prev_query_alpha", "prev_tether")):
                a.forget_static_conf = None
            a.forget_map_version = 0
            a._forget_role_idx = None             # role index cache (per map version) cleared per video
            a.forget_prev_dense = None            # H1: no map yet -> newest latent read whole
            a.forget_rand_seed = int(getattr(self, "renoise_seed", 0) or 0)   # random-selection control
            if f.get("sticky"):
                a._fg_sticky = {}                 #: sticky ranked kept sets, per video

    def _forget_update(self, clean_latents):
        """After each block's refresh: append the student's OWN clean latents to a window of the last window_latents;
        at the first block boundary at or after each due time warmup_s + j * map_refresh_s (video seconds = generated
        latents * 0.25 s) re-estimate the role map from that window and install it on every layer (shuffled with a
        fixed seeded permutation for the control). Never reads an evaluation mask."""
        f = self.forget
        if f["roles"] != "online_var":
            return
        x = clean_latents.detach()
        if x.shape[0] != 1:
            raise ValueError("factorial.forget supports batch size 1")
        for i in range(x.shape[1]):
            self._forget_buf.append(x[0, i])
        self._forget_buf = self._forget_buf[-f["window_latents"]:]
        self._forget_n += int(x.shape[1])
        refresh_l = max(1, int(round(f["map_refresh_s"] / f["latent_s"])))
        due = f["warmup_latents"] + self._forget_j * refresh_l
        if self._forget_n < due:
            return
        while f["warmup_latents"] + self._forget_j * refresh_l <= self._forget_n:
            self._forget_j += 1                                  # one estimate per boundary, however many dues passed
        log_var, thr, material = self._volatility_roles(torch.stack(self._forget_buf), f["threshold"])
        real = material.clone()
        if self._forget_perm is not None:
            material = material[self._forget_perm.to(material.device)]
        version = len(self.forget_events) + 1
        dense = None
        frozen = bool(f.get("prev_freeze")) and getattr(self, "_prev_dense_frozen", False)
        if f.get("prev_split"):
            # H1: moving OR uncertain (within prev_band log10 units below the threshold) -> exact newest tokens
            dense = log_var.flatten() > (thr - float(f["prev_band"]))
            if self._forget_perm is not None:
                dense = dense[self._forget_perm.to(dense.device)]
        static_conf = None
        static_on = any(k in f for k in ("prev_query_alpha", "prev_tether"))
        if static_on and dense is not None and not frozen:
            from wan21.modules.static_fidelity import static_confidence
            static_conf = static_confidence(log_var, thr, dense, taper=0.3)
        for a in self._attn_modules():
            a.forget_role = material.clone()
            a.forget_map_version = version
            a._forget_role_idx = None
            if dense is not None and not frozen:
                a.forget_prev_dense = dense.clone()
                if static_conf is not None:
                    a.forget_static_conf = static_conf.clone()
        if dense is not None and f.get("prev_freeze"):
            self._prev_dense_frozen = True
        hw = tuple(log_var.shape)
        self.forget_events.append(dict(version=version, n_latents=self._forget_n, t_s=self._forget_n * f["latent_s"],
                                       due_s=due * f["latent_s"], window_latents=len(self._forget_buf), threshold=thr,
                                       material_fraction=float(material.float().mean()), shuffled=self._forget_perm is not None,
                                       prev_dense_fraction=None if dense is None else float(dense.float().mean()),
                                       prev_dense_installed=dense is not None and not frozen,
                                       material=material.view(hw).cpu(),
                                       material_unshuffled=real.view(hw).cpu() if self._forget_perm is not None else None))
        if static_on:
            conf = self.generator.model.blocks[0].self_attn.forget_static_conf
            self.forget_events[-1].update(
                static_confidence=None if conf is None else conf.view(hw).cpu(),
                static_conf_installed=static_conf is not None,
                static_conf_share=None if conf is None else float((conf > 0).float().mean()),
                static_conf_mean=None if conf is None else float(conf.float().mean()))
        print(f"[forget] map v{version} at {self._forget_n * f['latent_s']:.2f} s (due {due * f['latent_s']:.2f} s) from the "
              f"last {len(self._forget_buf)} latents: material {100 * float(material.float().mean()):.1f} % "
              f"(threshold log10 var {thr:.3f}){' SHUFFLED' if self._forget_perm is not None else ''}")

    def _role_routing_reset(self):
        self._role_sum = self._role_sumsq = None
        self._role_n = 0
        self.role_routing_map = None
        if self.role_routing is not None:
            for a in self._attn_modules():
                a.source_role = None

    def _role_routing_update(self, clean_latents):
        """Accumulate the student's OWN clean latents (denoised_pred of each block, [B, F, C, H, W])
        and, at the first block boundary with >= warmup_latents generated latents, freeze the role
        map: per-token temporal variance of the latent (over the generated latents so far, mean
        over channels) on the latent grid (60 x 104), mean-pooled over the DiT patch (1, 2, 2) to
        the attention token grid (30 x 52 = 1560 tokens, row-major h * W_tok + w, the order of the
        patch embedding's flatten), log10, thresholded (Otsu, or a quantile) into HIGH temporal
        variation (field `material`, False in `support`) vs LOW temporal variation (field `support`,
        True). The classes are variation levels of the student's own output, not a scene / material
        identity, and C3 routes QUERIES between two complete source entries (low-variation queries
        read the sharp entry at every step; high-variation queries the scheduled one). Optional
        uncertain band (`uncertain`, half-width in log10 units around the threshold): tokens inside
        it are neither class and follow the plain C2 schedule (same reads as high-variation).
        Optional control (`shuffle`): the class labels are spatially permuted with a permutation
        seeded by the video seed (identical class proportions, frozen at the same block boundary).
        Frozen afterwards (never updated: the map must not follow drift it is meant to prevent) and
        written to the output folder by inference.py as <video>_routing.png + .json, compared with
        the frozen evaluation masks only AFTER the fact."""
        rr = self.role_routing
        if rr is None or self.role_routing_map is not None:
            return
        x = clean_latents.detach().float()
        if x.shape[0] != 1:
            raise ValueError("role_routing supports batch size 1")
        s = x[0].sum(0)                                        # [C, H, W]
        s2 = (x[0] ** 2).sum(0)
        self._role_sum = s if self._role_sum is None else self._role_sum + s
        self._role_sumsq = s2 if self._role_sumsq is None else self._role_sumsq + s2
        self._role_n += int(x.shape[1])
        if self._role_n < rr["warmup_latents"]:
            return
        n = float(self._role_n)
        var = (self._role_sumsq / n - (self._role_sum / n) ** 2).clamp_min(0).mean(0)   # [H, W]
        ps = getattr(self.generator.model, "patch_size", (1, 2, 2))
        tok = torch.nn.functional.avg_pool2d(var[None, None], kernel_size=(int(ps[1]), int(ps[2])))[0, 0]
        log_var = torch.log10(tok + 1e-12)
        flat = log_var.flatten()
        if flat.numel() != self.frame_seq_length:
            raise RuntimeError(f"role map has {flat.numel()} tokens, attention has {self.frame_seq_length}")
        if rr["threshold"] == "auto":
            thr = self._otsu_threshold(flat)
        else:
            thr = float(torch.quantile(flat, float(rr["threshold"])))
        band = float(rr.get("uncertain_band", 0.0))
        # labels: 0 = low temporal variation (`support`: sharp entry at every step), 1 = high temporal
        # variation (`material`: the scheduled entry), 2 = uncertain band (scheduled entry, like 1)
        labels = torch.full_like(flat, 1, dtype=torch.long)
        labels[flat <= thr - band] = 0
        if band > 0:
            labels[(flat > thr - band) & (flat <= thr + band)] = 2
        labels_real = labels.clone()
        shuffled = bool(rr.get("shuffle", False))
        if shuffled:
            seed = getattr(self, "renoise_seed", None)
            if seed is None:
                raise RuntimeError("role_routing.shuffle needs the video seed (pipeline.renoise_seed, set by "
                                   "inference.py under factorial.latent_indexed_renoise)")
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + 7)
            perm = torch.randperm(flat.numel(), generator=g).to(labels.device)
            labels = labels[perm]                               # same class counts, random positions
        support = labels == 0
        material = labels == 1
        uncertain = labels == 2
        self.role_routing_map = dict(support=support.view_as(log_var).cpu(), log_var=log_var.cpu(),
                                     labels=labels.view_as(log_var).cpu(),
                                     threshold=float(thr), n_latents=self._role_n,
                                     support_fraction=float(support.float().mean()),
                                     material_fraction=float(material.float().mean()),
                                     uncertain_fraction=float(uncertain.float().mean()),
                                     uncertain_band=band, shuffled=shuffled,
                                     rule="otsu" if rr["threshold"] == "auto" else f"quantile{rr['threshold']}",
                                     classes="support = low temporal variation (sharp entry every step); "
                                             "material = high temporal variation (scheduled entry); "
                                             "uncertain = within the band (scheduled entry)")
        if shuffled:
            self.role_routing_map["support_unshuffled"] = (labels_real == 0).view_as(log_var).cpu()
            self.role_routing_map["labels_unshuffled"] = labels_real.view_as(log_var).cpu()
        for a in self._attn_modules():
            a.source_role = support.clone()
        print(f"[role_routing] map frozen after {self._role_n} generated latents "
              f"({self._role_n / rr['latent_fps']:.2f} s): support {100 * self.role_routing_map['support_fraction']:.1f} % "
              f"material {100 * self.role_routing_map['material_fraction']:.1f} % "
              f"threshold(log10 var)={thr:.3f} rule={self.role_routing_map['rule']}"
              + (f" uncertain {100 * self.role_routing_map['uncertain_fraction']:.1f} % (band +-{band})" if band > 0 else "")
              + (" SHUFFLED control (labels permuted, seeded by the video seed)" if shuffled else "")
              + " [support = low, material = high temporal variation of the student's own latents]")

    def _block_plan(self, block_sizes):
        """[(frames, warm)] for the denoising loop. Default: only block 0 is warm."""
        K = getattr(self, "warm_start_frames", None)
        K_ns = getattr(self, "warm_start_frames_nosplit", None)
        if K is None and K_ns is None:
            return [(n, i == 0) for i, n in enumerate(block_sizes)]
        if K_ns is not None:       # ws6: whole blocks, warm iff the block starts before K
            plan, done = [], 0
            for n in block_sizes:
                plan.append((n, done < K_ns))
                done += n
            return plan
        plan, done = [], 0
        for n in block_sizes:
            if done >= K:
                plan.append((n, False))
            elif done + n <= K:
                plan.append((n, True))
            else:   # straddles K: warm head, then the cold rest of the same block
                plan.append((K - done, True))
                plan.append((done + n - K, False))
            done += n
        return plan

    def _latent_indexed_noise(self, like, start_frame, timestep):
        """Re-noise field for latents [start_frame, start_frame + F) towards `timestep`, one
        generator per (renoise_seed, absolute latent, nominal timestep): partition-invariant."""
        if self.renoise_seed is None:
            raise RuntimeError("latent_indexed_renoise needs pipeline.renoise_seed (inference.py)")
        if like.shape[0] != 1:
            raise ValueError("latent_indexed_renoise supports batch size 1")
        t_key = int(round(float(timestep)))
        frames = []
        for j in range(like.shape[1]):
            g = torch.Generator(device=like.device)
            g.manual_seed(int(self.renoise_seed) * 10_000_019 + (start_frame + j) * 1_009 + t_key)
            frames.append(torch.randn(like.shape[2:], generator=g, device=like.device,
                                      dtype=like.dtype))
        return torch.stack(frames, dim=0)[None]

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf = False,
        report_timing: bool = False,
        region_freeze: Optional[torch.Tensor] = None,
        region_cfg: Optional[dict] = None,
        color_cfg: Optional[dict] = None,
        anchor_cfg: Optional[dict] = None,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.alg_enabled and initial_latent is None:
            raise ValueError("ALG requires an I2V initial_latent reference frame")
        if getattr(self, "source_schedule", None) is not None or getattr(self, "dymos_ref_scale", None) is not None:
            if initial_latent is None or not self.independent_first_frame or initial_latent.shape[1] != 1:
                raise ValueError("factorial.source_schedule / dymos_ref_scale need the one-latent I2V "
                                 "source (independent_first_frame with initial_latent[:, :1])")
        if getattr(self, "clean_sigma", None) is not None and (
                region_freeze is not None or (color_cfg is not None and color_cfg.get("enabled"))
                or (anchor_cfg is not None and anchor_cfg.get("enabled"))):
            raise ValueError("factorial.clean_sigma emits the refresh latent: not with region / colour / anchor corrections")
        if getattr(self, "role_routing", None) is not None or (
                getattr(self, "forget", None) is not None and self.forget["roles"] == "online_var"):
            if region_freeze is not None or (anchor_cfg is not None and anchor_cfg.get("mask") is not None):
                raise ValueError("factorial.role_routing / forget (online map) are model-intrinsic: no region / anchor mask "
                                 "may be passed to inference()")
        # Derive tokens-per-latent-frame from the ACTUAL latent shape (patchify stride),
        # overriding the 60x104 default. Wrong stride -> KV-cache index/width mismatch
        # (e.g. 58x104 needs 29*52=1508, not 30*52=1560).
        _ps = getattr(self.generator.model, "patch_size", (1, 2, 2))
        self.frame_seq_length = (height // _ps[1]) * (width // _ps[2])
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        # Guard: with global attention (local_attn_size == -1) there is no KV
        # eviction, but the cache is a fixed 32760 tokens (== 21 latent frames).
        # Long videos that exceed that would otherwise overflow mid-run with a
        # cryptic width-0 slice error. Fail early with an actionable message.
        if self.local_attn_size == -1:
            max_global_frames = 21  # global cache holds 21 latent frames (21 * frame_seq_length)
            if num_output_frames > max_global_frames:
                raise ValueError(
                    f"local_attn_size=-1 (global attention) supports at most "
                    f"{max_global_frames} latent frames, but {num_output_frames} were "
                    f"requested. The rolling KV cache is not active — your config's "
                    f"`model_kwargs` (e.g. local_attn_size: 21, sink_size: 3) are not "
                    f"being applied to the model. Check that the run loads "
                    f"causal_forcing_dmd_f2s_steady.yaml and that model_kwargs reach "
                    f"WanDiffusionWrapper."
                )

        # Optional: start generation timer (excludes VAE decode). Only runs
        # when the caller explicitly opts in.
        if report_timing:
            torch.cuda.synchronize()
            self._gen_start_time = time.time()

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # Native-i2v conditioning: CLIP image features (constant across the rollout)
        # ride in conditional_dict; the per-block `y` (image-latent concat) is sliced
        # to each block's frames inside the loop below. T2V leaves both None.
        if clip_fea is not None:
            conditional_dict["clip_fea"] = clip_fea

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        for block in self.generator.model.blocks:
            attn = getattr(block, "self_attn", None)
            if attn is not None and hasattr(attn, "reset_echo_memory"):
                attn.reset_echo_memory()
        self._phase1_attention.set_denoise_step_fraction(None)
        if getattr(self, "role_routing", None) is not None:
            self._role_routing_reset()
        if getattr(self, "forget", None) is not None:
            self._forget_reset()
        if any(getattr(self, n_, None) is not None for n_ in ("forcing_kv", "future_kv", "tether_cache")):
            # ports: per-video state (Future Forcing alive / merged tokens, attention history, query buffer;
            # TetherCache memory store; Forcing-KV shared band selection)
            for a_ in self._attn_modules():
                if getattr(a_, "future_kv", None) is not None:
                    a_._ff_state = dict(cur=None, alive={}, acc={}, cnt={}, qbuf=[], last={})
                if getattr(a_, "tether_cache", None) is not None:
                    a_._tc_state = dict(seeded=False, mem_k=None, mem_v=None, mem_ids=[], next=None, events=[])
                if getattr(a_, "forcing_kv", None) is not None:
                    a_.forcing_kv["shared"].clear()
        if getattr(self, "source_schedule", None) is not None:
            for a in self._attn_modules():            # entries of the previous video, if any
                a.source_entries = None
                a.source_read = "sharp"
                a.source_role = None
        if getattr(self, "context_accounting", False):
            #: layer 0 records one row per generated block (logging only)
            self.context_rows = []
            self.context_static_info = self._context_static()
            self._attn_modules()[0].context_log = self.context_rows
        g_alt = self._guidance_setup(text_prompts, conditional_dict, batch_size, noise)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                if self.alg_enabled:
                    self._prepare_alg_reference_banks(
                        initial_latent[:, :1], conditional_dict, timestep)
                if getattr(self, "source_schedule", None) is not None:
                    self._prepare_source_entries(initial_latent[:, :1], conditional_dict)
                if getattr(self, "forget", None) is not None and self.forget.get("reference") == "xref":
                    self._prepare_xref_reference(conditional_dict)
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        block_plan = self._block_plan(all_num_frames)
        for block_index, (current_num_frames, warm_block) in enumerate(tqdm.tqdm(block_plan)):
            # Optional: time the first block (TTFC). Excludes the KV-cache
            # refresh pass that follows the main denoising.
            if report_timing and block_index == 0:
                torch.cuda.synchronize()
                _first_block_start = time.time()

            if profile:
                block_start.record()

            # Optional periodic KV flush (semantic purification). Runs at block
            # start, only after one full local window has formed (warmup), on a
            # fixed block period. No-op unless kv_flush_period > 0.
            if (self.kv_flush_period > 0 and block_index > 0
                    and current_start_frame >= self.local_attn_size
                    and block_index % self.kv_flush_period == 0):
                self._kv_flush()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Native-i2v: slice the conditioning latent `y` to exactly the frames this
            # block generates (frame 0 carries the input image, later frames are masked).
            # Misalignment here = identity smear, so it tracks current_start_frame.
            if y is not None:
                conditional_dict["y"] = y[:, current_start_frame:current_start_frame + current_num_frames]

            # Select denoising schedule: block 0 may use a dedicated schedule
            # when provided by the config; otherwise all blocks share the same list.
            current_denoising_list = (
                self.denoising_step_list_first_chunk
                if warm_block and self.denoising_step_list_first_chunk is not None
                else self.denoising_step_list
            )

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(current_denoising_list):
                step_fraction = index / max(1, len(current_denoising_list))
                self._phase1_attention.set_denoise_step_fraction(step_fraction)
                if getattr(self, "source_schedule", None) is not None or getattr(self, "dymos_ref_scale", None) is not None:
                    self._set_source_step(block_index, warm_block, index, len(current_denoising_list))
                if self.alg_enabled:
                    use_lowpass = (index >= self.alg_clean_delay_steps
                                   and step_fraction < self.alg_step_ratio)
                    self._install_alg_reference_bank(use_lowpass)
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
                if g_alt is not None:
                    # prompt extrapolation: read-only alternative branch, same noisy input / past
                    denoised_pred = self._guidance_extrapolate(
                        denoised_pred, noisy_input, g_alt, conditional_dict, timestep, current_start_frame,
                        index, current_denoising_list)

                # TTC-style image anchor (training-free, inference-only): pull the STATIC
                # region of the CLEAN prediction toward the input image, then let the next
                # step re-noise + assimilate it. Operating on x0 (not noisy_input) avoids the
                # fresh-noise granularity; region-gating keeps the fluid region free (no
                # motion tug-of-war). No-op when anchor_cfg is None/disabled.
                denoised_pred = self._anchor_correct(
                    denoised_pred, anchor_cfg, current_timestep,
                    index, len(current_denoising_list))

                if index < len(current_denoising_list) - 1:
                    next_timestep = current_denoising_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        self._latent_indexed_noise(
                            denoised_pred, current_start_frame, next_timestep).flatten(0, 1)
                        if getattr(self, "latent_indexed_renoise", False) else
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])

            # Region-forcing (static-drift control): pull the static background
            # toward a reference, leaving the masked flow region free. `corrected`
            # is what we save/decode; `denoised_pred` is the model's own latent.
            corrected = denoised_pred
            if region_freeze is not None and initial_latent is not None:
                _rcfg = region_cfg or {}
                if _rcfg.get("reference", "anchor") == "prev":
                    # Drift-relative: pull toward the previous committed frame.
                    ref = output[:, current_start_frame - 1:current_start_frame]
                else:
                    ref = initial_latent[:, :1]  # anchor: input image (frame 0)
                corrected = self._region_correct(denoised_pred, ref, region_freeze, _rcfg)

            # Global color anchor (static-drift control, color/illumination only):
            # pull the per-channel DC / low-frequency of the WHOLE frame toward the
            # input image. Touches only the color band, so it holds color identity
            # without constraining motion texture (unlike a spatial freeze). Output
            # only — never written to the cache.
            if color_cfg is not None and color_cfg.get("enabled") and initial_latent is not None:
                if color_cfg.get("reference", "anchor") == "prev" and current_start_frame > 0:
                    cref = output[:, current_start_frame - 1:current_start_frame]
                else:
                    cref = initial_latent[:, :1]
                corrected = self._color_anchor_correct(corrected, cref, color_cfg)

            # Step 3.2: record the (corrected) output
            output[:, current_start_frame:current_start_frame + current_num_frames] = corrected

            # Record first-chunk latency (denoising only, before KV cache refresh).
            if report_timing and block_index == 0:
                torch.cuda.synchronize()
                self.first_chunk_time = time.time() - _first_block_start
                print(f"First chunk time: {self.first_chunk_time:.2f}s")

            # Step 3.3: rerun with timestep zero to update KV cache using clean context.
            # By default the cache is refreshed with the model's OWN latent (not the
            # corrected one) to avoid feeding it an out-of-distribution spliced latent
            # that would compound artifacts. --region_freeze_cache opts into the old
            # behavior (cache the corrected latent -> stronger but riskier).
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            cache_input = corrected if (region_cfg is not None and region_cfg.get("write_cache")) else denoised_pred

            self._phase1_attention.set_denoise_step_fraction(None)
            if getattr(self, "source_schedule", None) is not None or getattr(self, "dymos_ref_scale", None) is not None:
                self._set_source_step(block_index, warm_block, None, None)   # refresh: sharp, DyMoS off
            if self.alg_enabled:
                self._install_alg_reference_bank(False)
            cs = getattr(self, "clean_sigma", None)
            if cs is not None:
                # UnStep C5: re-noise x0 at sigma*, keep the t = 0 label, write the cache from x_s
                # and emit x0' = x_s - sigma* * flow (the eps draw is latent-indexed under latent_indexed_renoise,
                # key 20 = 1000 * sigma*, never used by the re-noise of a denoising step)
                eps = (self._latent_indexed_noise(denoised_pred, current_start_frame, 20)
                       if getattr(self, "latent_indexed_renoise", False) else torch.randn_like(denoised_pred))
                xs = ((1.0 - cs) * denoised_pred.float() + cs * eps.float()).to(denoised_pred.dtype)
                flow, _ = self.generator(
                    noisy_image_or_video=xs,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                output[:, current_start_frame:current_start_frame + current_num_frames] = \
                    (xs.float() - cs * flow.float()).to(output.dtype)
            else:
                self.generator(
                    noisy_image_or_video=cache_input,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            if getattr(self, "role_routing", None) is not None:
                self._role_routing_update(denoised_pred)      # the model's OWN clean latents
            if getattr(self, "forget", None) is not None:
                self._forget_update(denoised_pred)            #: the model's OWN clean latents

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if rectified_tf:
            mean = torch.load('laboratory/mean.pt').to(output.device)
            std = torch.load('laboratory/std.pt').to(output.device)
            noise = torch.randn_like(output).to(output.device)
            output -= mean

        # Record diffusion time (excluding VAE decode).
        if report_timing:
            torch.cuda.synchronize()
            self.last_generation_time = time.time() - self._gen_start_time

        if getattr(self, "packcache_decay", None) is not None or getattr(self, "fp_pool", None) is not None \
                or getattr(self, "forget", None) is not None:
            #: one line per video with the attended token counts (layer 0, first evaluation of
            # every generated block); the window would hold min(cur + k, 21) * frame_seq_length tokens
            st = list(self._attn_modules()[0].pack_stats)
            name = ("packcache_decay" if self.packcache_decay is not None else
                    "fp_pool" if self.fp_pool is not None else "forget")
            cfg = (self.packcache_decay if self.packcache_decay is not None else
                   self.fp_pool if self.fp_pool is not None else self.forget)
            fs = self.frame_seq_length
            W = self._attn_modules()[0].max_attention_size // fs
            if st:
                print(f"[factorial] {name} {({k: v for k, v in cfg.items() if k != 'budget_tokens_by_distance'})}: "
                      f"attended tokens per block: first {st[0]}, last {st[-1]}, mean {sum(st) / len(st):.0f}, "
                      f"max {max(st)} (full 21-frame window {W * fs}; {len(st)} blocks)")
            self.pack_token_stats = st
            for a in self._attn_modules():
                a.pack_stats = []
        if g_alt is not None:
            rec = self.guidance_record
            rel = [e["rms_diff"] / max(e["rms_base"], 1e-12) for e in rec["evals"]]
            print(f"[factorial] guidance {rec['branch']} s={rec['scale']}: {rec['cost']['alt_forwards']} alternative + "
                  f"{rec['cost']['base_denoise_forwards']} base denoising forwards, {rec['cost']['refresh_forwards']} "
                  f"refreshes; ||z_base - z_alt|| / ||z_base|| mean {sum(rel) / max(1, len(rel)):.4f} "
                  f"max {max(rel) if rel else 0:.4f}")

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        # In-place denorm/clamp: a fresh (video*0.5+0.5).clamp() allocates a 2nd full
        # pixel tensor (~18 GB for a 240s / 3840-frame clip) and OOMs on long horizons.
        # Numerically identical to the out-of-place form, so shorter clips are unaffected.
        video = video.mul_(0.5).add_(0.5).clamp_(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    @staticmethod
    def _alg_lowpass(x, factor):
        """Official ALG downsample/upsample filter in latent space."""
        import torch.nn.functional as F
        b, f, c, h, w = x.shape
        low_h, low_w = max(1, round(h / factor)), max(1, round(w / factor))
        flat = x.reshape(b * f, c, h, w).float()
        flat = F.interpolate(flat, size=(low_h, low_w), mode="bilinear", align_corners=False)
        flat = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False)
        return flat.reshape_as(x).to(x.dtype)

    def _snapshot_reference_bank(self):
        fs = self.frame_seq_length
        return [(cache["k"][:, :fs].clone(), cache["v"][:, :fs].clone())
                for cache in self.kv_cache1]

    def _install_alg_reference_bank(self, lowpass):
        bank = self._alg_lowpass_bank if lowpass else self._alg_clean_bank
        if bank is None:
            return
        fs = self.frame_seq_length
        for cache, (key, value) in zip(self.kv_cache1, bank):
            cache["k"][:, :fs].copy_(key)
            cache["v"][:, :fs].copy_(value)

    def _prepare_alg_reference_banks(self, reference, condition, timestep):
        """Build clean/low-pass reference K/V once, then switch banks per step."""
        self._alg_clean_bank = self._snapshot_reference_bank()
        filtered = self._alg_lowpass(reference, self.alg_factor)
        self.generator(
            noisy_image_or_video=filtered,
            conditional_dict=condition,
            timestep=timestep * 0,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=0,
        )
        self._alg_lowpass_bank = self._snapshot_reference_bank()
        self._install_alg_reference_bank(False)

    @staticmethod
    def _latent_lowpass(x, sigma):
        """Gaussian low-pass over the spatial (H, W) dims of a latent
        [B, F, C, H, W]. Done in float32 for stability, returned in x's dtype."""
        import torchvision.transforms.functional as TF
        b, f, c, h, w = x.shape
        k = int(2 * math.ceil(2 * sigma) + 1)
        xb = x.reshape(b * f, c, h, w).float()
        xb = TF.gaussian_blur(xb, kernel_size=[k, k], sigma=[sigma, sigma])
        return xb.reshape(b, f, c, h, w).to(x.dtype)

    def _anchor_correct(self, pred, cfg, current_timestep, index, n_steps):
        """TTC-style training-free image anchor on the CLEAN prediction.

        Softly pulls `pred` toward the input-image latent, optionally only in the
        static region (so the fluid region stays free) and only at low-noise steps.
        Correcting x0 (not the noisy input) and letting the loop re-noise/assimilate
        avoids the granularity of mixing fresh noise into noisy_input; region-gating
        avoids the global-anchor motion tug-of-war.

        cfg keys: enabled, strength (w in [0,1]), latent [B,1,C,H,W],
          mask [1,1,1,H,W] (1=anchor/static, 0=free/fluid; None=global anchor),
          tmax (anchor only steps with timestep<=tmax), skip_last (skip final step).
        """
        if cfg is None or not cfg.get("enabled"):
            return pred
        w = max(0.0, min(1.0, float(cfg.get("strength", 0.0))))
        if w <= 0.0:
            return pred
        tmax = cfg.get("tmax", None)
        if tmax is not None and int(current_timestep) > int(tmax):
            return pred
        if cfg.get("skip_last", False) and index == n_steps - 1:
            return pred
        img = cfg["latent"][:, :1].to(pred.dtype)   # [B,1,C,H,W] -> broadcasts over frames
        mask = cfg.get("mask", None)                # [1,1,1,H,W] static-anchor weight, or None
        gate = w if mask is None else (w * mask.to(pred.dtype))
        return pred + gate * (img - pred)

    def _color_anchor_correct(self, pred, ref, cfg):
        """Global color/illumination stabilizer toward `ref` (the input image).

        Shifts only the per-channel DC (spatial mean), or a large-sigma low-pass
        field, of the *whole* frame toward the reference. Motion lives in the
        spatial/temporal variation, which is untouched — so `strength` can be high
        (up to 1.0 = exact per-frame DC match) without flattening the flow.

        pred: [B, F, C, H, W]; ref: [B, 1, C, H, W]; cfg: strength, sigma.
        """
        beta = float(cfg.get("strength", 0.5))
        sigma = float(cfg.get("sigma", 0.0))
        if sigma and sigma > 0:
            ref_c = self._latent_lowpass(ref, sigma)
            pred_c = self._latent_lowpass(pred, sigma)
        else:
            ref_c = ref.mean(dim=(-1, -2), keepdim=True)    # [B,1,C,1,1]
            pred_c = pred.mean(dim=(-1, -2), keepdim=True)   # [B,F,C,1,1]
        return pred + beta * (ref_c - pred_c)

    def _region_correct(self, pred, ref, freeze, cfg):
        """Pull the static region of `pred` toward reference `ref`.

        pred   : [B, F, C, H, W] model latent for the current block
        ref    : [B, 1, C, H, W] anchor (frame 0) or previous committed frame
        freeze : [1, 1, 1, H, W] in [0, 1] (1 => fully corrected)
        cfg    : dict with strength, lowfreq, lowfreq_sigma

        - strength (alpha): soft pull instead of a hard overwrite.
        - lowfreq: correct only the low-frequency (color/illumination) component,
          keeping the model's own high-frequency texture; best for color drift.
        """
        strength = float(cfg.get("strength", 1.0)) if cfg else 1.0
        eff = (strength * freeze).clamp(0, 1)
        if cfg and cfg.get("lowfreq"):
            sigma = float(cfg.get("lowfreq_sigma", 1.5))
            lp_ref = self._latent_lowpass(ref, sigma)
            lp_pred = self._latent_lowpass(pred, sigma)
            # shift only the low-frequency band toward the reference
            return pred + eff * (lp_ref - lp_pred)
        return eff * ref + (1.0 - eff) * pred

    def _kv_flush(self):
        """Compact every layer's self-attention KV cache to
        ``[frame-0 anchor | last K frames]`` and discard the middle context.

        Used for periodic "semantic purification" of long rollouts: the static
        frame-0 anchor is preserved and the last ``kv_flush_keep_frames`` frames
        provide a smooth motion tail, while accumulated middle context (where
        drift/artifacts build up) is dropped so the window re-grows from a clean
        state. ``global_end_index`` is deliberately left untouched so the model's
        local/global delta bookkeeping continues to line up on the next block.

        Only valid with relativistic_rope (raw-K cache). Cross-attention is left
        alone — the text prompt is unchanged, so re-encoding it would be a no-op.
        """
        fs = self.frame_seq_length
        anchor = fs                                   # frame 0 (hard anchor)
        keep = self.kv_flush_keep_frames * fs
        for i in range(len(self.kv_cache1)):
            c = self.kv_cache1[i]
            local_end = int(c["local_end_index"].item())
            # Nothing to purify yet: cache hasn't grown past anchor + tail.
            if local_end <= anchor + keep:
                continue
            # .clone() the source so the (anchor, tail) regions can't alias.
            c["k"][:, anchor:anchor + keep] = c["k"][:, local_end - keep:local_end].clone()
            c["v"][:, anchor:anchor + keep] = c["v"][:, local_end - keep:local_end].clone()
            c["local_end_index"].fill_(anchor + keep)


    def _attn_shape(self):
        """(num_heads, head_dim) of the loaded DiT.

        The KV caches were allocated as [B, T, 12, 128] -- Wan2.1-1.3B's 1536-dim / 12
        heads. Wan2.2-TI2V-5B is 3072-dim / 24 heads, so a fixed 12 silently mis-shapes
        the cache and the first write fails with a size mismatch (or worse, would alias
        heads if the product happened to match).
        """
        m = self.generator.model
        nh = int(getattr(m, "num_heads", 12))
        dim = int(getattr(m, "dim", nh * 128))
        return nh, dim // nh

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.echo_enabled:
            # Echo owns its bounded A/C/R state inside attention; this compatibility
            # cache only carries scalar indices and one frame of scratch storage.
            kv_cache_size = self.frame_seq_length
        elif self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Global attention: cache 21 latent frames' worth of tokens (shape-derived,
            # 32760 for 60x104). frame_seq_length is set from the real latent in inference().
            kv_cache_size = 21 * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, *self._attn_shape()], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, *self._attn_shape()], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, *self._attn_shape()], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, *self._attn_shape()], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
