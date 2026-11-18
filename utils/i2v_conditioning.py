"""Native-i2v conditioning: build the CLIP image features (`clip_fea`) and the
conditioning latent (`y`) that a Wan-I2V model concatenates onto the noise channels.

Mirrors the bidirectional teacher in wan/image2video.py (L207-246): frame 0 carries
the input image, the rest are masked; a 4-channel temporal mask is concatenated to
the 16-channel image latent -> 20 channels (so in_dim becomes 16+20=36).

Layout here is [B, F, C, H, W] (matching the noise tensor used everywhere in the
causal pipeline), NOT the teacher's channels-first [C, F, H, W].
"""
import os
import torch

_I2V_DIR = "../ckpt/wan_models/Wan2.1-I2V-14B-480P"
_CLIP_CKPT = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"


def load_clip(device, i2v_dir=_I2V_DIR, dtype=torch.float16):
    """Load the Wan2.1 CLIP-H image encoder (weights ship inside the I2V-14B dir)."""
    # Imported here, not at module level: wan21.modules needs a CUDA device at import
    # time, while build_y / build_svi_y are pure tensor code that CPU unit tests use.
    from wan21.modules.clip import CLIPModel
    return CLIPModel(
        dtype=dtype, device=device,
        checkpoint_path=os.path.join(i2v_dir, _CLIP_CKPT),
        tokenizer_path=os.path.join(i2v_dir, "xlm-roberta-large"),
    )


def build_y(image_latent_f0, num_latent_frames, anchor_period=0):
    """Build the i2v conditioning latent `y` from the frame-0 image latent.

    Unified across train and inference: frame 0 carries the image latent + mask=1;
    the rest are zeros + mask=0. The student learns its conditioning from scratch,
    so this exact (and cheap) representation is what matters — as long as train and
    inference use the SAME builder.

    anchor_period > 0 (INFERENCE-ONLY probe): also re-inject the image latent + mask
    at frames 0, P, 2P, ... to combat long-rollout drift (the frame-0 anchor fades as
    the KV window rolls past it). P = num_frame_per_block re-anchors each block; P=1
    every frame. NOTE this is OOD vs Stage-1 (trained frame-0-only) — a diagnostic to
    see if persistent anchoring reduces 60s fBD; if it helps, it belongs in Stage-3.

    image_latent_f0 : [B, 1, 16, h, w]   (frame-0 VAE latent of the input image)
    returns y       : [B, F, 20, h, w]   (4-ch temporal mask + 16-ch image latent)
    """
    b, _, c, h, w = image_latent_f0.shape
    F = int(num_latent_frames)
    y_lat = image_latent_f0.new_zeros(b, F, c, h, w)
    mask = image_latent_f0.new_zeros(b, F, 4, h, w)
    anchor_frames = range(0, F, anchor_period) if anchor_period and anchor_period > 0 else [0]
    for f in anchor_frames:
        y_lat[:, f] = image_latent_f0[:, 0]
        mask[:, f] = 1.0
    return torch.cat([mask, y_lat], dim=2)  # [B, F, 20, h, w]


def build_svi_y(source_latent, num_latent_frames, motion_latent=None):
    """SVI Pro's conditioning latent `y`, in the [B, F, C, H, W] layout.

    Mirrors diffsynth WanVideoUnit_ImageEmbedderVAE for the SVI Pro pipeline
    (wan_video_svi_pro.py:505-531): slot 0 = anchor (source) latent, slots 1..M = the
    motion latents of the previous clip when given, zeros elsewhere; the 4-channel
    temporal mask is 1 on slot 0 ONLY (the mask marks the anchor, not the motion
    latents -- SVI builds `msk[:, 1:] = 0` before concatenating).

      C1 (first clip):   y = [mask 1 | VAE(source), zeros]            motion_latent=None
      C2 (continuation): y = [mask 1 | anchor, mask 0 | L_p, zeros]   motion_latent=L_p

    With motion_latent=None this is exactly build_y(source_latent, F) (slot 0 only).

    source_latent : [B, 1, 16, h, w]
    motion_latent : [B, M, 16, h, w] or None, 1 + M <= num_latent_frames
    returns y     : [B, F, 20, h, w]
    """
    b, n, c, h, w = source_latent.shape
    if n != 1:
        raise ValueError(f"build_svi_y: source_latent must hold one latent, got {n}")
    F = int(num_latent_frames)
    y_lat = source_latent.new_zeros(b, F, c, h, w)
    mask = source_latent.new_zeros(b, F, 4, h, w)
    y_lat[:, 0] = source_latent[:, 0]
    mask[:, 0] = 1.0
    if motion_latent is not None:
        m = motion_latent.shape[1]
        if motion_latent.shape[0] != b or motion_latent.shape[2:] != (c, h, w):
            raise ValueError(
                f"build_svi_y: motion_latent {tuple(motion_latent.shape)} does not match "
                f"source_latent {tuple(source_latent.shape)}")
        if 1 + m > F:
            raise ValueError(
                f"build_svi_y: 1 anchor + {m} motion latents exceed {F} latent frames")
        y_lat[:, 1:1 + m] = motion_latent.to(y_lat.dtype)
        # mask stays 0 on the motion slots (SVI semantics)
    return torch.cat([mask, y_lat], dim=2)  # [B, F, 20, h, w]


@torch.no_grad()
def build_i2v_conditioning(image_pix, vae, clip, num_latent_frames, anchor_period=0):
    """Inference-side: encode the input image to its frame-0 latent + CLIP features,
    then build `y` via build_y (same builder used in training).

    image_pix : [C, H, W] in [-1, 1] at the model's pixel resolution.
    clip_fea  : [1, 257, 1280]   y : [1, F_lat, 20, h, w]
    anchor_period : 0 = frame-0-only (Stage-1 default); >0 = persistent-anchor probe.
    """
    vae_dtype = next(vae.model.parameters()).dtype
    img_in = image_pix[None, :, None].to(dtype=vae_dtype)        # [1, C, 1, H, W]
    image_latent_f0 = vae.encode_to_latent(img_in)              # [1, 1, 16, h, w]
    clip_fea = clip.visual([image_pix[:, None, :, :]])          # [1, 257, 1280]
    y = build_y(image_latent_f0, num_latent_frames, anchor_period=anchor_period)
    return clip_fea, y
