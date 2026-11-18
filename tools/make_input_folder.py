#!/usr/bin/env python3
"""Build an input folder for scripts/run_scenetether.sh from images and text prompts.

    python tools/make_input_folder.py --out inputs/my_scenes --pair photo1.jpg "prompt one" --pair photo2.png "prompt two"
    python tools/make_input_folder.py --out inputs/my_scenes --list pairs.json   # [{"image": ..., "prompt": ...}, ...]

Each image is center-cropped to the model's 832x480 aspect ratio, resized to 832x480 (Lanczos) and saved as PNG.
Prompts that describe a fixed camera and name what stays still and what moves work best (see examples/).
"""
import argparse
import json
from pathlib import Path

from PIL import Image

W, H, RATIO = 832, 480, '26-15'


def crop_resize(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    if w / h > W / H:
        nw = round(h * W / H); x = (w - nw) // 2; im = im.crop((x, 0, x + nw, h))
    else:
        nh = round(w * H / W); y = (h - nh) // 2; im = im.crop((0, y, w, y + nh))
    return im.resize((W, H), Image.Resampling.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pair', nargs=2, action='append', metavar=('IMAGE', 'PROMPT'), default=[])
    ap.add_argument('--list', help='JSON list of {"image": path, "prompt": text}')
    a = ap.parse_args()
    pairs = [tuple(p) for p in a.pair]
    if a.list:
        pairs += [(e['image'], e['prompt']) for e in json.loads(Path(a.list).read_text())]
    if not pairs:
        ap.error('give --pair or --list')
    out = Path(a.out); (out / RATIO).mkdir(parents=True, exist_ok=True)
    meta = []
    for i, (img, prompt) in enumerate(pairs):
        name = f'{i:03d}_{Path(img).stem}.png'
        crop_resize(img).save(out / RATIO / name)
        meta.append(dict(file_name=name, caption=prompt.strip(),
                         target_crop=dict(target_bbox=[0, 0, W, H], target_ratio=RATIO),
                         type='user', origin_width=W, origin_height=H))
    (out / f'target_crop_info_{RATIO}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'{len(meta)} inputs written to {out}')


if __name__ == '__main__':
    main()
