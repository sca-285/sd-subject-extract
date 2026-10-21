import launch

# BiRefNet's model code (loaded from Hugging Face with trust_remote_code) needs
# these three on top of what the WebUI already ships.
for package in ("timm", "kornia", "einops"):
    if not launch.is_installed(package):
        launch.run_pip(f"install {package}", f"Subject Extract requirement: {package}")
