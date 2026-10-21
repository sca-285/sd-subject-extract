"""Main-subject extraction: mask, transparent cutout, and the background with
the subject removed. Pure torch / numpy / OpenCV; no WebUI imports, so it can
be tested on its own.

The mask comes from BiRefNet (dichotomous image segmentation): it separates
the dominant foreground - a person, an animal, a product, a car - from
everything else, with fine hair and edge detail.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODELS = {
    # name shown in the UI: (Hugging Face repo, input size)
    "BiRefNet (general, best)": ("ZhengPeng7/BiRefNet", 1024),
    "BiRefNet matting (soft hair / fur)": ("ZhengPeng7/BiRefNet-matting", 1024),
    "BiRefNet HR (2048, large images)": ("ZhengPeng7/BiRefNet_HR", 2048),
    "BiRefNet lite (fast, less VRAM)": ("ZhengPeng7/BiRefNet_lite", 1024),
}
DEFAULT_MODEL = "BiRefNet (general, best)"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_model = None
_model_name = None


# ------------------------------------------------------------------ model

def _load(name, device):
    """The model for `name`, on `device`. Loaded once, then reused."""
    global _model, _model_name
    repo, _ = MODELS.get(name, MODELS[DEFAULT_MODEL])
    if _model is None or _model_name != name:
        from transformers import AutoModelForImageSegmentation
        print(f"[Subject Extract] Loading {repo} (downloaded on first use)...")
        _model = None
        m = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        _model, _model_name = m.eval(), name
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    _model.to(device=device, dtype=dtype)
    return _model, dtype


def park():
    """Move the model to system RAM between jobs (instead of re-reading it)."""
    if _model is not None:
        _model.to("cpu", dtype=torch.float32)


def unload():
    global _model, _model_name
    _model = _model_name = None


@torch.no_grad()
def predict_mask(image: Image.Image, model_name=DEFAULT_MODEL, device="cpu", model=None):
    """Foreground probability, float32 (H, W) in 0..1, at the image's own size.

    `model` lets tests pass a stand-in; normally the named BiRefNet is used.
    """
    _, size = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    if model is None:
        model, dtype = _load(model_name, device)
    else:
        dtype = torch.float32
    rgb = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    x = (x - torch.tensor(_MEAN).view(1, 3, 1, 1)) / torch.tensor(_STD).view(1, 3, 1, 1)
    out = model(x.to(device=device, dtype=dtype))
    logits = out[-1] if isinstance(out, (list, tuple)) else out
    prob = logits.float().sigmoid()
    prob = F.interpolate(prob, size=(image.height, image.width), mode="bilinear", align_corners=False)
    return prob[0, 0].clamp(0, 1).cpu().numpy()


# ------------------------------------------------------------------ mask shaping

def _shift(mask, px):
    """Grow (px > 0) or shrink (px < 0) a soft mask by |px| pixels."""
    if px == 0:
        return mask
    t = torch.from_numpy(mask)[None, None]
    k = 2 * abs(int(px)) + 1
    if px > 0:
        t = F.max_pool2d(t, k, stride=1, padding=abs(int(px)))
    else:
        t = -F.max_pool2d(-t, k, stride=1, padding=abs(int(px)))
    return t[0, 0].numpy()


def _feather(mask, px):
    if px <= 0:
        return mask
    import cv2
    k = int(px) * 4 + 1
    return cv2.GaussianBlur(mask, (k, k), px)


def _largest_only(mask):
    """Keep the biggest connected object; drop stray specks and islands."""
    import cv2
    binary = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:                                   # background + at most one object
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == keep).astype(np.float32)
    # A little slack around it so its own soft edge survives.
    region = _shift(region, 3)
    return mask * region


def refine(mask, hard=False, edge_shift=0, feather=0, largest_only=False):
    """Shape the raw probability into the mask that gets used."""
    m = mask.astype(np.float32)
    if largest_only:
        m = _largest_only(m)
    if hard:
        m = (m > 0.5).astype(np.float32)
    m = _shift(m, edge_shift)
    m = _feather(m, feather)
    return np.clip(m, 0.0, 1.0)


# ------------------------------------------------------------------ outputs

def cutout(image: Image.Image, mask) -> Image.Image:
    """The subject on a transparent background (RGBA, straight alpha)."""
    rgba = image.convert("RGB").copy()
    rgba.putalpha(Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L"))
    return rgba


def mask_image(mask) -> Image.Image:
    """White subject on black, 8-bit."""
    return Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L")


def _push_pull(rgb, known):
    """Fill unknown pixels from their surroundings, coarse to fine.

    Push: halve the image repeatedly, averaging only known pixels, until the
    hole is covered. Pull: walk back up, filling each level's unknown pixels
    from the smoothly upsampled level below. The result is a soft, streak-free
    continuation of the surroundings (Telea-style inpainting drags edge
    colours across a large hole in straight streaks instead).
    """
    img = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    w = torch.from_numpy(known)[None, None].float()
    levels = []
    while True:
        levels.append((img, w))
        if min(img.shape[-2:]) <= 2 or bool((w > 0).all()):
            break
        s_img = F.avg_pool2d(img * w, 2, ceil_mode=True)
        s_w = F.avg_pool2d(w, 2, ceil_mode=True)
        img, w = s_img / s_w.clamp_min(1e-6), (s_w > 0).float()
    filled = levels[-1][0]
    for img, w in reversed(levels[:-1]):
        up = F.interpolate(filled, size=img.shape[-2:], mode="bilinear", align_corners=False)
        filled = img * w + up * (1 - w)
    return filled[0].permute(1, 2, 0).numpy()


def remove_subject(image: Image.Image, mask, grow=12) -> Image.Image:
    """The background with the subject taken out and the hole filled in.

    `grow` widens the hole first so the subject's edge glow and any halo go
    with it. The fill is a smooth continuation of the surroundings, not
    generative inpainting: a small subject disappears cleanly, a subject that
    fills most of the frame leaves a soft blurred patch - send the background
    and the mask to img2img inpaint for a detailed plate.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    hole = _shift((mask > 0.5).astype(np.float32), grow)
    if hole.max() == 0:
        return image.convert("RGB")
    filled = _push_pull(rgb, (hole < 0.5).astype(np.float32))
    blend = _feather(hole, max(2, grow // 3))[..., None]
    out = rgb * (1 - blend) + filled * blend
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


OUTPUTS = ("Subject (transparent PNG)", "Mask", "Background (subject removed)")


def extract(image: Image.Image, mask, wanted=OUTPUTS, grow=12):
    """{output name: image} for the outputs asked for."""
    out = {}
    if OUTPUTS[0] in wanted:
        out[OUTPUTS[0]] = cutout(image, mask)
    if OUTPUTS[1] in wanted:
        out[OUTPUTS[1]] = mask_image(mask)
    if OUTPUTS[2] in wanted:
        out[OUTPUTS[2]] = remove_subject(image, mask, grow=grow)
    return out
