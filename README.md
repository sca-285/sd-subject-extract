# Stable Diffusion Subject Extract

Separate a main subject from its background in **Forge-family WebUIs** using BiRefNet. The subject can be a person, animal, product, vehicle or another prominent foreground object.

## Outputs

- **Subject PNG:** the extracted subject on a transparent background.
- **Mask PNG:** a greyscale mask, with the subject in white and the background in black.
- **Background PNG:** the scene with the subject removed and the resulting area filled from nearby pixels.

The background fill is a smooth continuation of the surroundings, not generative inpainting. When the subject occupies most of the frame, the filled area may look like a soft patch. For a clean plate, use the extracted mask with an inpainting workflow.

## Install

Copy this folder into `extensions/sd-subject-extract/` and restart the WebUI. The included installer handles the required Python packages when they are missing. BiRefNet weights are downloaded from Hugging Face on first use.

## Use it

### After txt2img or img2img

1. Enable **Subject Extract** in the generation panel.
2. Choose a model and select the outputs you need.
3. Generate as usual.

The extension runs after the other image-processing extensions. Results can be shown in the gallery and saved to a `subject-extract` folder next to the WebUI's output folders.

### From PNG Info

Drop an image into **PNG Info** and use the **Subject Extract** panel. If that WebUI does not provide a suitable panel there, the extension creates a separate **Subject Extract** tab. PNG Info results are saved under `outputs/subject-extract` by default.

The saved files can retain the source image's generation parameters as PNG text metadata. Turn off **Keep the image's generation parameters** to omit them. The source seed is included in the output filenames when available, and the results note whether generation parameters were present.

## Models

| Model | When to use it |
|---|---|
| **BiRefNet (general, best)** | Default choice for general subject extraction. |
| **BiRefNet matting (soft hair / fur)** | Try for softer boundaries such as hair or fur. |
| **BiRefNet HR (2048, large images)** | Higher-resolution input for detailed or large images; uses more resources. |
| **BiRefNet lite (fast, less VRAM)** | A lighter option when speed or memory matters more. |

The model is kept in system RAM between runs. The available models trade speed, memory use and edge detail, so check the cutout at its intended output size.

## Testing

Run `python tests_se.py` from the extension folder to exercise the processing path with a stand-in for BiRefNet. The test does not need to download the model or use a GPU.
