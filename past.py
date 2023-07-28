# https://gist.github.com/Sentdex/130c225d90acec7c808b8ba5aba0eda1
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os

post=""
prompts=[
   "Capture the essence of nostalgia as you step into a bygone era in a quaint desert village of Dubai. The wide-angle shot captures the rustic charm of the village, nestled amidst the golden dunes of the desert. Ancient mud-brick buildings with weathered walls stand as a testament to the passage of time. Date palm trees sway gently in the breeze, casting shadows over the sandy paths. In the distance, the mesmerizing sight of camels grazing and traditional wind towers adorning the skyline transport you back in time. The warm tones and soft vignette add an old vintage touch, evoking memories of a simpler and authentic past.",
"Transport yourself to a hidden gem in the heart of Dubai's desert, where time seems to stand still. The wide-angle shot captures the enchanting beauty of an old village tucked away amidst the rolling sand dunes. The sun-kissed, weathered walls of the traditional mud-brick houses exude a timeless charm. Palm-frond roofs and wooden doors tell stories of the village's rich heritage. The golden glow of the setting sun casts a warm hue over the scene, evoking a sense of nostalgia. As you explore the dusty streets, the view extends to the vast desert expanse, creating a picturesque sight that blends the old world with the timeless beauty of nature.",
"Experience the allure of an old vintage village in Dubai, frozen in time, yet steeped in history. The wide-angle shot captures the captivating beauty of traditional adobe houses, adorned with intricate patterns and ornate wooden balconies. The village's charm is further accentuated by the surrounding desert landscape, with windswept dunes stretching endlessly into the horizon. As you wander through the alleys, you'll encounter local artisans practicing their crafts, adding authenticity to the scene. The rich, earthy tones and soft focus lend an old vintage aesthetic, transporting you to a bygone era where traditions and simplicity reigned."
]
opath = "output/old"
samples = 10
os.makedirs(opath,exist_ok=True)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                revision="fp16",
                                                torch_dtype=torch.float16, 
                                                use_auth_token=True)
pipe = pipe.to("cuda")
for ind,prompt in enumerate(prompts):
    os.makedirs(os.path.join(opath,str(ind)),exist_ok=True)
    for cnt in range(samples):
        seed = random.randint(0, 10000)
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(prompt+post, num_inference_steps=100, generator=generator).images[0]
        image.save(os.path.join(opath,str(ind),f"{seed}.png"))
    # break