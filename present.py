# https://gist.github.com/Sentdex/130c225d90acec7c808b8ba5aba0eda1
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os

post=""
prompts=[
   "Behold the awe-inspiring grandeur of Dubai's cityscape in this wide-angle shot. The vibrant city lights illuminate the night sky, creating a breathtaking panorama of towering skyscrapers. The iconic Burj Al Arab stands proudly amidst the sparkling waters of the Arabian Gulf, its sail-shaped silhouette adding a touch of elegance to the scene. In the distance, the Burj Khalifa pierces the sky, a testament to human achievement and innovation. The city's futuristic architecture seamlessly blends with the warmth of the desert, creating a captivating visual symphony of modernity and tradition. The hyper-realistic details and dynamic colors transport you into a world of wonder and possibility.",
"Capture the essence of Dubai's dynamic energy in this wide-angle cityscape shot. The sun sets on the horizon, casting a golden glow over the city's iconic landmarks. The Burj Khalifa, the world's tallest building, stands tall, reaching for the heavens. Nearby, the sail-shaped Burj Al Arab reflects the hues of the setting sun, adding a touch of luxury and splendor. The city's impressive skyline features an array of gleaming skyscrapers, each a testament to Dubai's remarkable growth and progress. The bustling city streets, lined with palm trees and bustling with activity, lend a sense of vibrancy and life to the scene. The hyper-realistic details and vibrant colors bring the city's iconic landmarks to life, creating an electrifying visual experience.",
"Embrace the futuristic allure of Dubai's cityscape in this wide-angle shot that feels almost otherworldly. The Burj Khalifa pierces the sky like a needle, towering over the modern metropolis below. The Burj Al Arab, resembling a majestic ship at sail, stands as a symbol of opulence and sophistication. A myriad of futuristic skyscrapers create an urban landscape that seems to defy gravity. The city's sparkling lights mirror the starry night sky, evoking a sense of wonder and excitement. As you immerse yourself in the hyper-realistic details and vivid colors of the scene, you'll find yourself transported to a mesmerizing world where dreams and reality seamlessly intertwine."
]
opath = "output/present"
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