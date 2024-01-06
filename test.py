import pyvips

# Load the image
image = pyvips.Image.new_from_file(
    "/pesgisipth/NDPI/H23-9432;S14;MSK1 - 2023-12-12 04.55.10.ndpi"
)
# Crop the image
# Parameters: left, top, width, height
cropped_image = image.crop(10000, 10000, 1000, 1000)
# Save the cropped image
cropped_image.write_to_file("region2.jpg")
