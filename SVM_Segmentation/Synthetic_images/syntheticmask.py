import os
import random
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

def syntheticmask():
    foreground = Image.new('RGBA', (200, 200))
    draw = ImageDraw.Draw(foreground)
    draw.ellipse((20, 20, 180, 180), fill='black', outline='black')
    background = Image.new('RGBA', (500, 500))

    # Scale the foreground
    scale = random.uniform(0.5, 1)  # Pick something between .5 and 1
    new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
    foreground = foreground.resize(new_size)

    # Choose a random x,y position for the foreground
    max_position = (background.size[0] - foreground.size[0], background.size[1] - foreground.size[1])
    if max_position[0] > 0 and max_position[1] > 0:
        paste_position = (random.randint(0, max_position[0]), random.randint(0, max_position[1]))

        # Extract the alpha channel from the foreground and paste it into a new image the size of the background
        alpha_mask = foreground.getchannel(3)
        new_alpha_mask = Image.new('L', background.size, color=0)
        new_alpha_mask.paste(alpha_mask, paste_position)

        # Grab the alpha pixels above a specified threshold
        alpha_threshold = 200
        mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
        mask = Image.fromarray(np.uint8(mask_arr) * 255, 'L')

        plt.imshow(mask, cmap='gray')
        plt.show()

        return mask

output_dir = '../Synthetic_images/GeneratedImages'
for i in range(10):
    mask = syntheticmask()
    mask.save(f'{output_dir}/mask{i}.png', 'PNG')
