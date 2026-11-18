"""Region-forcing masks (#4): author by hand, or auto-detect the flow region.

A mask is a grayscale image where WHITE (255) marks the DYNAMIC flow region
(water/fire/smoke/rain/...) the model should generate, and BLACK (0) marks the
STATIC background that gets frozen to the input image. It is resized to the
latent grid (60x104) at inference time, so exact resolution does not matter.

Two ways to obtain masks:

1) Automatic (used by inference.py when --region_forcing is set): CLIPSeg
   text-prompt segmentation. `generate_flow_mask_clipseg()` segments the flow
   region from keywords found in the prompt. inference.py saves the result named
   after each input image so you can inspect / hand-edit and re-run.

2) Manual box (this script's CLI), for a quick rectangle:
   python make_region_mask.py --size 832x480 --box x1 y1 x2 y2 --out masks/river.png

Polarity matches inference.py's default (white = flow). If you instead paint the
static area white, pass --region_mask_invert to inference.py.
"""
import argparse
import os
import re

from PIL import Image, ImageDraw


# Flow / natural-motion nouns we segment for. Extend freely.
FLOW_KEYWORDS = [
    "flowing water", "water", "river", "stream", "canal", "sea", "ocean",
    "wave", "waves", "tide", "ripples", "current", "flood", "waterfall",
    "fountain", "splash", "spray", "fire", "flame", "flames", "smoke",
    "rain", "snow", "snowfall", "cloud", "clouds", "lava", "fog", "mist",
    "steam", "petals", "falling petals", "blossoms", "leaves", "falling leaves",
    "lava", "rising smoke", "steam", "explosion", "debris",
    "volcano eruption", "dust", "sandstorm", "moving clouds", "smoke plume",
    "wind", "wind gust", "windy", "breeze", "leaves blowing", "flags waving",
    "storm", "hurricane", "tornado", "cyclone", "whirlwind", "vortex",
    "rain", "snow", "snowfall", "hail", "sleet", "blizzard", "drizzle",
    "heavy rainfall", "light rain", "thunderstorm", "lightning", "thunder",
    "flow", "moving", "motion", "dynamic", "dynamics", "moving clouds", "moving water",
    "moving fire", "moving smoke", "moving rain", "moving snow", "moving leaves",
    "moving petals", "moving debris", "moving lava", "moving fog", "moving steam",
]


def extract_flow_queries(prompt):
    """Pick the flow nouns mentioned in the prompt (whole-word match).

    Falls back to a generic flow set when the prompt names none, so CLIPSeg
    always has something sensible to look for.
    """
    p = (prompt or "").lower()
    found = []
    seen = set()
    for kw in FLOW_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", p) and kw not in seen:
            found.append(kw)
            seen.add(kw)
    return found or ["flowing water", "fire", "smoke", "rain", "moving clouds"]


# Lazily-loaded CLIPSeg model (small, ~150MB), cached across calls.
_CLIPSEG = {}


def _stub_torchao_quantizer():
    """Prevent transformers from importing torchao while loading CLIPSeg.

    transformers' quantizer auto-import eagerly loads `torchao`, and torchao
    0.16 is incompatible with torch 2.5.x (it references `torch.int1..int7`,
    which don't exist) so the whole import chain crashes — even though we never
    quantize anything. We pre-register a harmless stub for the torchao quantizer
    submodule so transformers imports CLIPSeg without ever touching torchao.

    No-op once torchao actually works (then transformers imports it normally and
    this stub is never consulted, since we only insert it if absent).
    """
    import sys
    import types
    modname = "transformers.quantizers.quantizer_torchao"
    if modname in sys.modules:
        return

    class _Stub(types.ModuleType):
        def __getattr__(self, name):  # any symbol transformers asks for -> dummy
            return object

    sys.modules[modname] = _Stub(modname)


def _get_clipseg(device):
    if "model" not in _CLIPSEG:
        import torch  # noqa: F401 (kept for clarity; transformers needs it)
        # NOTE: do NOT call _stub_torchao_quantizer() — torchao is uninstalled in this
        # stack, so transformers skips it via is_torchao_available(). The stub poisons
        # sys.modules and breaks torchvision's import-time inspect walk
        # ("type object 'object' has no attribute 'endswith'"). Plain import works now.
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        name = "CIDAS/clipseg-rd64-refined"
        _CLIPSEG["proc"] = CLIPSegProcessor.from_pretrained(name)
        _CLIPSEG["model"] = CLIPSegForImageSegmentation.from_pretrained(name).to(device).eval()
    return _CLIPSEG["proc"], _CLIPSEG["model"]


def generate_flow_mask_clipseg(pil_image, queries, device, threshold=0.4, dilate=0):
    """Segment the flow region of `pil_image` for the given text `queries`.

    Returns a PIL 'L' mask at the image's resolution where WHITE = flow region.
    `threshold` binarizes the combined CLIPSeg probability (set 0 to keep a soft
    mask); `dilate` (in output pixels) grows the flow region so real motion at
    the edges is not accidentally frozen.
    """
    import torch
    import torch.nn.functional as F

    proc, model = _get_clipseg(device)
    W, H = pil_image.size

    inputs = proc(
        text=list(queries), images=[pil_image] * len(queries),
        return_tensors="pt", padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # [N, h, w] (or [h, w] when N == 1)
    if logits.dim() == 2:
        logits = logits[None]
    probs = torch.sigmoid(logits.float())          # [N, h, w]
    flow = probs.amax(dim=0, keepdim=True)[None]    # combine queries -> [1,1,h,w]
    flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)

    if threshold and threshold > 0:
        flow = (flow >= threshold).float()
        if dilate and dilate > 0:
            k = int(2 * dilate + 1)
            flow = F.max_pool2d(flow, kernel_size=k, stride=1, padding=dilate)

    arr = (flow[0, 0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="L")


def main():
    ap = argparse.ArgumentParser(description="Manual box mask (for CLIPSeg auto masks, use inference.py --region_forcing).")
    ap.add_argument("--size", type=str, required=True, help="WxH of the mask, e.g. 832x480")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="White rectangle (flow region) in pixel coords.")
    ap.add_argument("--out", type=str, required=True, help="Output mask path (.png)")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    mask = Image.new("L", (w, h), 0)  # black = static background
    if args.box:
        ImageDraw.Draw(mask).rectangle(args.box, fill=255)  # white = flow region
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mask.save(args.out)
    print(f"wrote {args.out} ({w}x{h}); white area = "
          f"{100 * (sum(mask.getdata()) / 255) / (w * h):.1f}% (flow region)")


if __name__ == "__main__":
    main()
