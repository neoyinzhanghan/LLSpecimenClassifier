import pyvips

pth = "/pesgisipth/NDPI/H23-852;S12;MSKW - 2023-06-15 16.42.50.ndpi"

# pth = "/media/hdd3/neo/PB_slides/H23-852;S12;MSKW - 2023-06-15 16.42.50.ndpi"

# Load the image
image = pyvips.Image.new_from_file(pth, level=0)

topview = pyvips.Image.new_from_file(pth, level=7)

print(image.width, image.height)
# Crop the image
# Parameters: left, top, width, height

print("resizing and cropping")
# resize the image to the size of the topview
image = image.resize(topview.width / image.width)
cropped_image = image.crop(0, 0, topview.width, topview.height)

# print the dimensions of the cropped image
print(cropped_image.width, cropped_image.height)

print("saving")
# Save the cropped image
cropped_image.write_to_file("region2.jpg")
