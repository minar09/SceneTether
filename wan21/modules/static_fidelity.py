"""Opt-in, read-only attention helpers for No cache writes."""
import math

import torch
import torch.nn.functional as F


def static_confidence(log_var, threshold, moving, taper=0.3):
    """Low-variance interior confidence; one-token 8-neighbour band abstains.

    No motion, no static, or non-finite input is an ambiguous partition, so abstain.
    This estimates confidence in early stationarity, not semantic rigidity.
    """
    if log_var.ndim != 2 or moving.numel() != log_var.numel() or taper <= 0:
        raise ValueError("static confidence needs aligned 2D variance and a positive taper")
    moving = moving.reshape_as(log_var).bool()
    if not math.isfinite(float(threshold)) or not torch.isfinite(log_var).all() \
            or not moving.any() or moving.all():
        return torch.zeros_like(log_var).flatten()
    # Two dilations distinguish distance 1 (band) from distance >=2 (interior).
    adjacent = F.max_pool2d(moving.float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
    c = ((float(threshold) - log_var.float()) / float(taper)).clamp(0, 1)
    return (c * (~adjacent)).flatten()


def packed_reference_indices(kept, cur, frame_seqlen, device):
    """Gather source + current at their existing positions, never re-RoPE them."""
    source, current, offset = [], [], 0
    for fid, count in kept:
        count = int(count)
        if fid == 0:
            source.append((offset, count))
        elif fid == cur:
            current.append((offset, count))
        offset += count
    if len(source) != 1 or len(current) != 1 or source[0][1] != frame_seqlen \
            or current[0][1] != frame_seqlen:
        raise RuntimeError("static-query read requires one full source and one full current frame")
    return torch.cat([torch.arange(o, o + n, device=device) for o, n in source + current])


def static_query_read(out, q, k, v, kept, cur, frame_seqlen, confidence, alpha, attention):
    """Only positive-weight query rows are changed; zero-weight rows remain bitwise."""
    if confidence is None or alpha == 0:
        return out
    if q.shape[1] != frame_seqlen or confidence.numel() != frame_seqlen:
        raise RuntimeError("static-query read supports one aligned latent per block")
    weight = confidence.to(device=q.device, dtype=torch.float32) * float(alpha)
    rows = (weight > 0).nonzero(as_tuple=True)[0]
    if not rows.numel():
        return out
    columns = packed_reference_indices(kept, cur, frame_seqlen, q.device)
    ref = attention(q.index_select(1, rows), k.index_select(1, columns), v.index_select(1, columns))
    old = out.index_select(1, rows)
    w = weight.index_select(0, rows).to(out.dtype)[None, :, None, None]
    result = out.clone()
    result.index_copy_(1, rows, old + w * (ref - old))
    return result


def tether_values(values, source, indices, moving, confidence, strength, band_strength):
    """Same-position value interpolation; keys/phases and persistent tensors untouched."""
    if source is None or source.shape != values.shape:
        raise RuntimeError("static value tether requires an aligned pinned source")
    beta = confidence.to(values.device).float() * float(strength)
    if band_strength is not None:
        band = (~moving.to(values.device)) & (confidence.to(values.device) == 0)
        beta = torch.where(band, float(band_strength), beta)
    beta = beta.index_select(0, indices).to(values.dtype)[None, :, None, None]
    recent = values.index_select(1, indices)
    anchor = source.index_select(1, indices)
    return torch.where(beta > 0, recent + beta * (anchor - recent), recent)
