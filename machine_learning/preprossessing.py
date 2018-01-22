from PIL import Image


def image_to_bitmap(filename, output_name):
    image = Image.open(filename)
    image.convert("RGBA")  # Convert this to RGBA if possible

    pixel_data = image.load()

    if image.mode == "RGBA":
        # If the image has an alpha channel, convert it to white
        # Otherwise we'll get weird pixels
        for y in range(image.size[1]):  # For each row ...
            for x in range(image.size[0]):  # Iterate through each column ...
                # Check if it's opaque
                if pixel_data[x, y][3] < 255:
                    # Replace the pixel data with the colour white
                    pixel_data[x, y] = (255, 255, 255, 255)

    # Resize the image thumbnail
    image.save("ola.bmp")


infile = "../bitmap_data/2 + 2 = 4.png"

image_to_bitmap(infile, "ola.bmp")

