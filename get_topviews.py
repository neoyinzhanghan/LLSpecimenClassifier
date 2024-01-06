import os
import pyvips

wsi_dir = "/pesgisipth/NDPI"
save_dir = "/media/hdd3/neo/all_topviews"

### Traverse through all .ndpi files in the wsi_dir with filename starting with H and S
### use pyvips to get the topview of each .ndpi file which is the slide image at the lowest magnification
### save the topview as .jpg file in save_dir

os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(wsi_dir):
    if file.startswith("H") or file.startswith("S") and file.endswith(".ndpi"):
        # Construct the full path for the WSI file
        wsi_path = os.path.join(wsi_dir, file)

        # Load the WSI file at the lowest magnification level
        wsi = pyvips.Image.new_from_file(wsi_path, level="highest")

        # Construct the save path, removing the original file extension
        file_name_without_extension = os.path.splitext(file)[0]
        save_path = os.path.join(save_dir, file_name_without_extension + ".jpg")

        # Save the resized image
        wsi.write_to_file(save_path)
