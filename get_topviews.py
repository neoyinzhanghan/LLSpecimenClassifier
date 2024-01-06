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

for file in tqdm(os.listdir(wsi_dir), desc="Getting topviews"):
    if file.startswith("H") or file.startswith("S") and file.endswith(".ndpi"):
        # use pyvips to get the topview of each .ndpi file
        wsi = pyvips.Image.new_from_file(os.path.join(wsi_dir, file), level=1)

        # save the topview as .jpg file in save_dir
        jpg_name = Path(file).stem + ".jpg"

        # print(pyvips.Image.get_fields(wsi))

        # image = wsi.crop(10000, 10000, 1000, 1000) 

        # sys.exit()
        
        # print the dimensions of the topview
        print(wsi.width, wsi.height)

        # wsi.write_to_file(os.path.join(save_dir, jpg_name))
