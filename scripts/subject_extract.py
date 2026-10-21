"""Subject Extract - pull the main person / object out of the background.

Two ways in:
  * an accordion in txt2img / img2img that processes every finished image,
  * a panel in the PNG Info tab, under the image, for any image dropped there:
    the generation parameters PNG Info reads from it are kept in the files
    it saves. (A tab of its own only if a WebUI has no PNG Info image.)
"""

import os
import sys
import traceback

import gradio as gr
from modules import devices, images, script_callbacks, scripts
from modules.ui_components import InputAccordion

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from lib_se import extract as X  # noqa: E402
from lib_se.guide import GUIDE, PNG_GUIDE, TAB_GUIDE  # noqa: E402

SUFFIX = {X.OUTPUTS[0]: "-subject", X.OUTPUTS[1]: "-mask", X.OUTPUTS[2]: "-background"}


def _outputs_root():
    """Where the tab saves: inside the output folder the WebUI is set to use.

    The old version always wrote to <webui>/outputs/subject-extract, which is
    the wrong place as soon as the output directories are set to anywhere else
    in Settings - the files were saved, just not where they were expected.
    """
    base = ""
    try:
        from modules.shared import opts
        base = opts.outdir_samples or os.path.dirname(opts.outdir_extras_samples or "")
    except Exception:
        pass
    if not base:
        try:
            from modules.paths_internal import data_path
        except Exception:
            data_path = os.getcwd()
        base = os.path.join(data_path, "outputs")
    return os.path.join(base, "subject-extract")


def _write_parameters(path, info):
    """Put the generation parameters into a saved PNG as a text chunk, the way the WebUI does."""
    from PIL import Image, PngImagePlugin
    with Image.open(path) as im:
        im.load()
        meta = PngImagePlugin.PngInfo()
        for key, value in (im.info or {}).items():
            if isinstance(value, str) and key != "parameters":
                meta.add_text(key, value)
        meta.add_text("parameters", info)
        im.save(path, pnginfo=meta)


def save_png(img, folder, basename, seed, prompt, info, **kwargs):
    """Save through the WebUI (its file names, folders and callbacks), then add the parameters.

    The parameters are not handed to save_image: some WebUIs (reForge) then also hide
    them in the alpha channel, which makes a transparent cutout opaque and cannot
    handle the greyscale mask at all.
    """
    saved = images.save_image(img, folder, basename, seed, prompt, extension="png", **kwargs)
    path = saved[0] if isinstance(saved, (tuple, list)) else saved
    if info and path and str(path).lower().endswith(".png") and os.path.isfile(path):
        _write_parameters(path, info)
    return path


def run_one(image, model, wanted, hard, edge_shift, feather, largest, grow):
    """Mask + requested outputs for one image. Returns {name: PIL image}."""
    raw = X.predict_mask(image, model, device=devices.device)
    mask = X.refine(raw, hard=hard, edge_shift=int(edge_shift), feather=float(feather),
                    largest_only=largest)
    return X.extract(image, mask, wanted, grow=int(grow))


def options_ui(prefix):
    """The controls both the accordion and the tab use, in the same order.
    One per line, nothing collapsed."""
    model = gr.Dropdown(label="Model", choices=list(X.MODELS), value=X.DEFAULT_MODEL,
                        elem_id=f"{prefix}_model",
                        info="General = best all-round, matting = finer hair, HR = large images, lite = fastest.")
    wanted = gr.CheckboxGroup(label="Outputs", choices=list(X.OUTPUTS), value=list(X.OUTPUTS),
                              elem_id=f"{prefix}_outputs")
    hard = gr.Checkbox(label="Hard edge", value=False, elem_id=f"{prefix}_hard",
                       info="Off keeps soft, semi-transparent hair and fur edges.")
    largest = gr.Checkbox(label="Largest object only", value=False, elem_id=f"{prefix}_largest",
                          info="Drop smaller separate pieces (stray specks, a second person).")
    edge_shift = gr.Slider(label="Edge shift (px)", minimum=-20, maximum=20, step=1, value=0,
                           elem_id=f"{prefix}_shift", info="- tighter / + looser around the subject.")
    feather = gr.Slider(label="Feather (px)", minimum=0, maximum=20, step=0.5, value=0,
                        elem_id=f"{prefix}_feather", info="Softens the cut edge.")
    grow = gr.Slider(label="Background: hole growth (px)", minimum=0, maximum=60, step=1, value=12,
                     elem_id=f"{prefix}_grow",
                     info="Removes a bit past the subject's edge before filling, so no outline is left.")
    return [model, wanted, hard, edge_shift, feather, largest, grow]


class Script(scripts.Script):
    sorting_priority = 400

    def title(self):
        return "Subject Extract"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with InputAccordion(False, label="Subject Extract", elem_id=f"se_enabled_{tab}") as enabled:
            gr.Markdown(GUIDE)
            show = gr.Checkbox(label="Show in results", value=True, elem_id=f"se_show_{tab}")
            save = gr.Checkbox(label="Save to disk", value=True, elem_id=f"se_save_{tab}",
                               info="Next to the images, in a subject-extract subfolder.")
            opts = options_ui(f"se_{tab}")
        return [enabled, show, save, *opts]

    # postprocess, not a per-image hook: it runs after every other extension
    # has finished (Digital Mastering, Easy Color Corrector, Optical Realism,
    # ADetailer...), so what gets extracted is the final picture.
    def postprocess(self, p, processed, enabled, show, save, model, wanted, hard,
                    edge_shift, feather, largest, grow):
        if not enabled or not wanted:
            return
        first = processed.index_of_first_image
        finals = processed.images[first:]
        folder = os.path.join(p.outpath_samples, "subject-extract")
        try:
            for i, image in enumerate(finals):
                outs = run_one(image, model, wanted, hard, edge_shift, feather, largest, grow)
                seed = processed.all_seeds[i] if i < len(processed.all_seeds) else None
                prompt = processed.all_prompts[i] if i < len(processed.all_prompts) else None
                info = processed.infotexts[first + i] if first + i < len(processed.infotexts) else None
                for name, img in outs.items():
                    if save:
                        save_png(img, folder, "", seed, prompt, info, p=p, suffix=SUFFIX[name])
                    if show:
                        processed.extra_images.append(img)
        except Exception as exc:
            traceback.print_exc()
            print(f"[Subject Extract] Failed: {exc}")
        finally:
            X.park()
            devices.torch_gc()


# ------------------------------------------------------------------ PNG Info

def _image_info(image):
    """The generation parameters written into an image (PNG text, JPEG / WebP EXIF), or ''."""
    info = None
    try:
        info, _ = images.read_info_from_image(image)
    except Exception:
        info = None
    if not info:
        info = (getattr(image, "info", None) or {}).get("parameters")
    if isinstance(info, bytes):
        info = info.decode("utf-8", errors="ignore")
    return str(info or "").strip()


def _seed_and_prompt(infotext):
    """Seed and prompt from parameters, for the file names; (None, None) when there are none."""
    if not infotext:
        return None, None
    parse = None
    for name in ("modules.infotext_utils", "modules.generation_parameters_copypaste"):
        try:
            parse = __import__(name, fromlist=["parse_generation_parameters"]).parse_generation_parameters
            break
        except Exception:
            continue
    if parse is None:
        return None, None
    try:
        params = parse(infotext)
    except Exception:
        return None, None
    seed = params.get("Seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    return seed, params.get("Prompt") or None


def extract_tab(image, model, wanted, hard, edge_shift, feather, largest, grow, save, keep_info=True):
    if image is None:
        return [], "Drop an image in first."
    if not wanted:
        return [], "Tick at least one output."
    infotext = _image_info(image)
    source = image.convert("RGB") if image.mode != "RGB" else image
    try:
        outs = run_one(source, model, wanted, hard, edge_shift, feather, largest, grow)
    except Exception as exc:
        traceback.print_exc()
        return [], f"Failed: {exc}"
    finally:
        X.park()
        devices.torch_gc()
    notes = []
    if save:
        root = os.path.abspath(_outputs_root())
        seed, prompt = _seed_and_prompt(infotext) if keep_info else (None, None)
        try:
            os.makedirs(root, exist_ok=True)
            for name, img in outs.items():
                # Always PNG: the subject needs its alpha channel.
                save_png(img, root, "subject-extract", seed, prompt, infotext if keep_info else None,
                         suffix=SUFFIX[name], save_to_dirs=False)
            notes.append(f"Saved to `{root}`")
            print(f"[Subject Extract] Saved {len(outs)} files to {root}")
        except Exception as exc:
            traceback.print_exc()
            notes.append(f"Could not save to `{root}`: {exc}")
    if infotext:
        notes.append("The image's generation parameters are kept in the saved files."
                     if save and keep_info else "The image has generation parameters (see above).")
    else:
        notes.append("The image has no generation parameters.")
    return list(outs.values()), " \n".join(notes)


_PANEL = {"image": None}


def pnginfo_panel(image):
    """Under PNG Info's image: extract from that same image."""
    with gr.Accordion("Subject Extract", open=True, elem_id="se_png_panel"):
        gr.Markdown(PNG_GUIDE)
        go = gr.Button("Extract subject", variant="primary", elem_id="se_png_go")
        gallery = gr.Gallery(label="Result", elem_id="se_png_gallery", columns=3, height=300)
        status = gr.Markdown("", elem_id="se_png_status")
        opts = options_ui("se_png")
        save = gr.Checkbox(label="Save to outputs/subject-extract", value=True, elem_id="se_png_save")
        keep = gr.Checkbox(label="Keep the image's generation parameters in the saved files", value=True,
                           elem_id="se_png_keep",
                           info="The parameters PNG Info shows go into the -subject, -mask and -background PNGs, "
                                "and the seed and prompt into their names.")
    go.click(extract_tab, [image, *opts, save, keep], [gallery, status])


def on_after_component(component, **kwargs):
    elem_id = kwargs.get("elem_id") or getattr(component, "elem_id", None)
    if elem_id != "pnginfo_image" or _PANEL["image"] is not None:
        return
    _PANEL["image"] = component
    try:
        pnginfo_panel(component)
    except Exception:
        traceback.print_exc()
        print("[Subject Extract] Could not add the panel to PNG Info; it gets a tab of its own instead.")
        _PANEL["image"] = False


# ------------------------------------------------------------------ fallback: a tab of its own

def on_ui_tabs():
    if _PANEL["image"]:
        return []          # in PNG Info
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown(TAB_GUIDE)
                image = gr.Image(label="Image", type="pil", elem_id="se_tab_image")
                opts = options_ui("se_tab")
                save = gr.Checkbox(label="Save to outputs/subject-extract", value=True,
                                   elem_id="se_tab_save")
                go = gr.Button("Extract", variant="primary", elem_id="se_tab_go")
            with gr.Column():
                gallery = gr.Gallery(label="Result", elem_id="se_tab_gallery", columns=3)
                status = gr.Markdown("")
        go.click(extract_tab, [image, *opts, save], [gallery, status])
    return [(tab, "Subject Extract", "subject_extract")]


script_callbacks.on_after_component(on_after_component)
script_callbacks.on_ui_tabs(on_ui_tabs)
