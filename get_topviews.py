import os
import pyvips
import sys
from tqdm import tqdm
from pathlib import Path

wsi_dir = "/pesgisipth/NDPI"
save_dir = "/media/hdd3/neo/all_topviews"

### Traverse through all .ndpi files in the wsi_dir with filename starting with H and S
### use pyvips to get the topview of each .ndpi file which is the slide image at the lowest magnification
### save the topview as .jpg file in save_dir

os.makedirs(save_dir, exist_ok=True)

ndpi_files = [
    file
    for file in os.listdir(wsi_dir)
    if file.startswith("H") or file.startswith("S") and file.endswith(".ndpi")
]

for i, file in enumerate(tqdm(ndpi_files)):
    if file.startswith("H") or file.startswith("S") and file.endswith(".ndpi"):
        print("loading")
        # use pyvips to get the topview of each .ndpi file
        wsi_topview = pyvips.Image.new_from_file(os.path.join(wsi_dir, file), level=7)
        wsi_full = pyvips.Image.new_from_file(os.path.join(wsi_dir, file), level=0)

        # save the topview as .jpg file in save_dir
        jpg_name = f"{i}.jpg"

        print("resizing")

        # Calculate scale factors for both dimensions
        scale_factor_width = wsi_topview.width / wsi_full.width
        scale_factor_height = wsi_topview.height / wsi_full.height

        # Choose the smaller scale factor to maintain aspect ratio
        scale_factor = min(scale_factor_width, scale_factor_height)

        # Resize the full image
        wsi_full_resized = wsi_full.resize(scale_factor)

        print("saving")
        wsi_full_resized.write_to_file(os.path.join(save_dir, jpg_name))
