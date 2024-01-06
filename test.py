import pyvips

# Load the image
image = pyvips.Image.new_from_file(
    "/pesgisipth/NDPI/H23-9432;S14;MSK1 - 2023-12-12 04.55.10.ndpi", level=7
)

print(image.width, image.height)
# Crop the image
# Parameters: left, top, width, height
cropped_image = image.crop(0, 0, image.width, image.height)

# print the dimensions of the cropped image
print(cropped_image.width, cropped_image.height)
# Save the cropped image
cropped_image.write_to_file("region2.jpg")
