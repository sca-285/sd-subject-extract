"""python tests_se.py  - no WebUI, no GPU, no download: BiRefNet is replaced by a
stand-in that 'finds' a known shape, so every step after the model is checked
exactly."""
import importlib.util
import os
import pathlib
import sys
import tempfile
import time
import types

import numpy as np
import torch
from PIL import Image
from skimage import data

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
fails = []


def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra != "" else ""))
    if not cond:
        fails.append(name)


from lib_se import extract as X  # noqa: E402

IMG = Image.fromarray(data.astronaut())          # 512 x 512
H, W = IMG.height, IMG.width


class FakeBiRefNet(torch.nn.Module):
    """Returns logits for a disc in the middle, the way BiRefNet returns a list."""
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x
        n = x.shape[-1]
        yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        disc = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 < (n / 4) ** 2).float()
        return [None, (disc * 20 - 10)[None, None]]


fake = FakeBiRefNet()
print("--- mask")
raw = X.predict_mask(IMG, X.DEFAULT_MODEL, model=fake)
ok("the model sees a 1024x1024 ImageNet-normalised input",
   tuple(fake.seen.shape) == (1, 3, 1024, 1024) and float(fake.seen.mean()) < 1.0 and float(fake.seen.min()) < 0)
ok("the mask comes back at the image's size, 0..1", raw.shape == (H, W) and raw.min() >= 0 and raw.max() <= 1)
ok("...and is the subject: 1 in the middle, 0 in the corner", raw[H // 2, W // 2] > 0.99 and raw[5, 5] < 0.01)
wide = Image.fromarray(np.zeros((300, 700, 3), np.uint8))
ok("non-square images keep their shape", X.predict_mask(wide, X.DEFAULT_MODEL, model=fake).shape == (300, 700))

print("\n--- shaping")
area = lambda m: float((m > 0.5).mean())
ok("edge shift + grows, - shrinks",
   area(X.refine(raw, edge_shift=8)) > area(raw) > area(X.refine(raw, edge_shift=-8)))
ok("hard edge is strictly 0/1", set(np.unique(X.refine(raw, hard=True))) <= {0.0, 1.0})
soft = X.refine(X.refine(raw, hard=True), feather=4)
ok("feather softens the edge", ((soft > 0.05) & (soft < 0.95)).sum() > ((raw > 0.05) & (raw < 0.95)).sum())
speck = raw.copy(); speck[10:30, 10:30] = 1.0
ok("largest object only drops a separate speck, keeps the subject",
   X.refine(speck, largest_only=True)[20, 20] == 0 and X.refine(speck, largest_only=True)[H // 2, W // 2] > 0.99)

print("\n--- outputs")
mask = X.refine(raw)
outs = X.extract(IMG, mask)
cut = outs[X.OUTPUTS[0]]
ok("three outputs by default", list(outs) == list(X.OUTPUTS))
ok("cutout: RGBA, alpha is the mask, colour untouched",
   cut.mode == "RGBA" and np.array_equal(np.asarray(cut)[..., 3], np.rint(mask * 255).astype(np.uint8))
   and np.array_equal(np.asarray(cut)[..., :3], np.asarray(IMG)))
ok("mask: 8-bit greyscale", outs[X.OUTPUTS[1]].mode == "L")
bg = np.asarray(outs[X.OUTPUTS[2]]).astype(int)
orig = np.asarray(IMG).astype(int)
inside = np.abs(bg - orig)[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20].mean()
outside = np.abs(bg - orig)[:40, :40].mean()
ok("background: the subject is gone, the rest untouched", inside > 20 and outside == 0,
   f"inside {inside:.1f}, far outside {outside:.1f}")
only = X.extract(IMG, mask, [X.OUTPUTS[1]])
ok("only the outputs asked for", list(only) == [X.OUTPUTS[1]])
big = IMG.resize((2048, 2048))
bmask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((2048, 2048))) / 255.0
t = time.perf_counter(); X.remove_subject(big, bmask); dt = time.perf_counter() - t
ok("background fill at 2048x2048 stays quick", dt < 5, f"{dt:.2f}s")
ok("an empty mask leaves the background as is",
   np.array_equal(np.asarray(X.remove_subject(IMG, np.zeros((H, W), np.float32))), np.asarray(IMG)))

print("\n--- in txt2img (fake WebUI)")


class _C:
    made = []

    def __init__(self, *a, **k):
        self.kw = k; self.events = []; _C.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def click(self, fn, inputs=None, outputs=None, **k):
        self.events.append((fn, inputs, outputs))
    change = blur = click


gr = types.ModuleType("gradio")
for n in ("Dropdown", "Button", "Slider", "Checkbox", "CheckboxGroup", "Markdown", "Row", "Column",
          "Accordion", "Image", "Gallery", "Blocks", "Tab", "Tabs", "Group"):
    setattr(gr, n, type(n, (_C,), {}))
sys.modules["gradio"] = gr
saved = []
mods = types.ModuleType("modules"); mods.__path__ = []
im = types.ModuleType("modules.images")
written = {}


def _fake_save(img, path, basename, seed=None, prompt=None, extension="png", info=None, p=None, suffix="", **k):
    os.makedirs(path, exist_ok=True)
    full = os.path.join(path, f"{basename}-{len(saved):04}-{seed}{suffix}.png")
    img.save(full)
    saved.append((path, seed, suffix, img.mode, full))
    return full, None


im.save_image = _fake_save
scr = types.ModuleType("modules.scripts"); scr.Script = type("Script", (), {}); scr.AlwaysVisible = object()
dev = types.ModuleType("modules.devices"); dev.device = torch.device("cpu"); dev.torch_gc = lambda: None
cb = types.ModuleType("modules.script_callbacks"); tabs = []; cb.on_ui_tabs = tabs.append; after = []; cb.on_after_component = after.append
uic = types.ModuleType("modules.ui_components"); uic.InputAccordion = type("InputAccordion", (_C,), {})
tmp = tempfile.mkdtemp()
pi = types.ModuleType("modules.paths_internal"); pi.data_path = tmp
sh = types.ModuleType("modules.shared")
sh.opts = types.SimpleNamespace(outdir_samples="", outdir_extras_samples=os.path.join(tmp, "out", "extras-images"))
for k, v in (("images", im), ("scripts", scr), ("devices", dev), ("script_callbacks", cb),
             ("ui_components", uic), ("paths_internal", pi), ("shared", sh)):
    setattr(mods, k, v); sys.modules["modules." + k] = v
sys.modules["modules"] = mods

spec = importlib.util.spec_from_file_location("se", ROOT / "scripts" / "subject_extract.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
real_predict = X.predict_mask
X.predict_mask = lambda image, name, device="cpu", model=None: real_predict(image, name, device, model=fake)

script = S.Script()
args = script.ui(False)
ok("accordion returns enabled, show, save + 7 options", len(args) == 10)
grid = Image.new("RGB", (64, 64))
a, b = IMG, IMG.transpose(Image.FLIP_LEFT_RIGHT)
processed = types.SimpleNamespace(images=[grid, a, b], index_of_first_image=1, all_seeds=[101, 102],
                                  all_prompts=["p1", "p2"], infotexts=["g", "i1", "i2"], extra_images=[])
p = types.SimpleNamespace(outpath_samples=os.path.join(tmp, "txt2img-images"))
opts = [X.DEFAULT_MODEL, list(X.OUTPUTS), False, 0, 0, False, 12]
script.postprocess(p, processed, True, True, True, *opts)
ok("every image except the grid is extracted, 3 files each", len(saved) == 6, len(saved))
ok("files carry the image's own seed and infotext",
   [s[1] for s in saved] == [101] * 3 + [102] * 3 and Image.open(saved[0][4]).info.get("parameters") == "i1"
   and Image.open(saved[5][4]).info.get("parameters") == "i2")
ok("the cutout stays transparent, the mask stays greyscale",
   Image.open(saved[0][4]).mode == "RGBA" and Image.open(saved[1][4]).mode == "L")
ok("...with -subject / -mask / -background suffixes into a subject-extract folder",
   [s[2] for s in saved[:3]] == ["-subject", "-mask", "-background"]
   and all(s[0].endswith("subject-extract") for s in saved))
ok("results are shown in the gallery", len(processed.extra_images) == 6)
saved.clear(); processed.extra_images.clear()
script.postprocess(p, processed, False, True, True, *opts)
ok("disabled does nothing", not saved and not processed.extra_images)
X.predict_mask = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model"))
script.postprocess(p, processed, True, True, True, *opts)
ok("a model failure is reported, not raised", not saved)
X.predict_mask = lambda image, name, device="cpu", model=None: real_predict(image, name, device, model=fake)

print("\n--- the tab")
saved.clear()
gallery, note = S.extract_tab(IMG, *opts, True)
ok("tab returns the three images and saves all three", len(gallery) == 3 and len(saved) == 3, len(saved))
ok("...into the WebUI's own output folder, not a fixed <webui>/outputs",
   all(s_[0] == os.path.join(tmp, "out", "subject-extract") for s_ in saved), saved[0][0] if saved else None)
ok("...with the three suffixes", [s_[2] for s_ in saved] == ["-subject", "-mask", "-background"])
ok("the note says where they went", "subject-extract" in note)
sh.opts.outdir_samples = os.path.join(tmp, "one-folder")
saved.clear()
S.extract_tab(IMG, *opts, True)
ok("a single output directory set in Settings is honoured",
   all(s_[0] == os.path.join(tmp, "one-folder", "subject-extract") for s_ in saved), saved[0][0] if saved else None)
sh.opts.outdir_samples = ""
saved.clear()
_real_save = im.save_image
im.save_image = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
gallery, note = S.extract_tab(IMG, *opts, True)
ok("a failed save is reported in the tab instead of failing silently",
   len(gallery) == 3 and "Could not save" in note, note)
im.save_image = _real_save
ok("saving off returns the images and says nothing about saving", "Saved" not in S.extract_tab(IMG, *opts, False)[1])
ok("tab without an image says so", S.extract_tab(None, *opts, True)[1].startswith("Drop"))

print("\n--- in PNG Info")
from PIL import PngImagePlugin  # noqa: E402
import io  # noqa: E402
meta = PngImagePlugin.PngInfo()
INFO = "an astronaut\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, Seed: 4242, Size: 512x512"
meta.add_text("parameters", INFO)
buf = io.BytesIO(); IMG.save(buf, "PNG", pnginfo=meta)
with_info = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")      # PNG Info hands over RGBA
parsed = types.ModuleType("modules.infotext_utils")
parsed.parse_generation_parameters = lambda text: {"Prompt": text.split("\n")[0], "Seed": "4242"}
sys.modules["modules.infotext_utils"] = parsed
saved.clear()
gallery, note = S.extract_tab(with_info, *opts, True, True)
ok("PNG Info's RGBA image is extracted like any other", len(gallery) == 3 and gallery[0].size == IMG.size)
ok("the image's parameters go into every saved file", len(saved) == 3 and all(Image.open(s_[4]).info.get("parameters") == INFO for s_ in saved))
ok("...and its seed into the names", all(s_[1] == 4242 for s_ in saved))
ok("the note says the parameters were kept", "kept in the saved files" in note, note)
saved.clear()
S.extract_tab(with_info, *opts, True, False)
ok("keeping the parameters can be turned off", all("parameters" not in Image.open(s_[4]).info and s_[1] is None for s_ in saved))
ok("an image without parameters says so", "no generation parameters" in S.extract_tab(IMG, *opts, False)[1])

ok("the panel waits for PNG Info's image", len(after) == 1)
_C.made.clear()
after[0](gr.Image(elem_id="txt2img_image"), elem_id="txt2img_image")
ok("...other images are left alone", not any(getattr(c, "kw", {}).get("elem_id") == "se_png_go" for c in _C.made))
png_image = gr.Image(elem_id="pnginfo_image")
after[0](png_image, elem_id="pnginfo_image")
go = next(c for c in _C.made if c.kw.get("elem_id") == "se_png_go")
fn, inputs, outputs = go.events[0]
ok("a panel under PNG Info's image extracts from that image", fn is S.extract_tab and inputs[0] is png_image and len(inputs) == 10 and len(outputs) == 2)
made = len(_C.made)
after[0](gr.Image(elem_id="pnginfo_image"), elem_id="pnginfo_image")
ok("...added once", len([c for c in _C.made if c.kw.get("elem_id") == "se_png_go"]) == 1)
ok("no tab of its own when the panel is in PNG Info", tabs[0]() == [])
S._PANEL["image"] = None
ok("a tab of its own when there is no PNG Info image", len(tabs[0]()) == 1)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
