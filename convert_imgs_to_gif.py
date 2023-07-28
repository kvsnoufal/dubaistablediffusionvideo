import os
from PIL import Image

def load_and_sort_png_images(folder_path):
    png_images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    png_images.sort(key=lambda x: int(os.path.splitext(x)[0][-5:]))
    return png_images

folder_path1 = "dubaitimeline/v1/v1_000000"
sorted_png_images1 = load_and_sort_png_images(folder_path1)
# print(sorted_png_images1)  # Verify that images are sorted correctly
folder_path2 = "dubaitimeline/v1/v1_000001"
sorted_png_images2 = load_and_sort_png_images(folder_path2)
folder_path3 = "dubaitimeline/v1/v1_000002"
sorted_png_images3 = load_and_sort_png_images(folder_path3)
# sorted_png_images = sorted_png_images1+sorted_png_images2

def save_as_gif(images_list,folder_path):
    gif_images = [Image.open(os.path.join(folder_path, img)) for img in images_list]
    # gif_images[0].save(gif_path, save_all=True, append_images=gif_images[1:], duration=500, loop=0)
    return gif_images
gif_images = save_as_gif(sorted_png_images1,folder_path1)+save_as_gif(sorted_png_images2,folder_path2)\
+save_as_gif(sorted_png_images3,folder_path3)

gif_path = "dubaitimeline/v1output.gif"
gif_images[0].save(gif_path, save_all=True, append_images=gif_images[1:], duration=500, loop=0)
