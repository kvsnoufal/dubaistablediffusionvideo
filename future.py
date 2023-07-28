# https://gist.github.com/Sentdex/130c225d90acec7c808b8ba5aba0eda1
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os

post=""
prompts=[
   "Capture a breathtaking wide-angle shot of a magnificent futuristic cityscape at dusk. Witness the towering skyscrapers adorned with vertical gardens, reflecting the golden hues of the setting sun. As you look up, marvel at the elegant dance of flying cars, weaving through the illuminated city skyline. In this metropolis of tomorrow, witness the fusion of nature and technology, with robots seamlessly coexisting alongside humans. The city's aura pulsates with vibrant colors, creating a visual masterpiece that will leave you in awe.",
"Step into the heart of a bustling futuristic city, where innovation knows no bounds. The city's vertical gardens sprawl across the skyscrapers, infusing the air with a sense of serenity amid the energetic hustle. As the sun casts its rays on the cityscape, witness the gleaming metal and glass facades that house advanced AI-driven robots working in harmony with humans. The streets are alive with the hum of flying cars soaring overhead, while holographic advertisements add a touch of futuristic charm. With every step, you'll feel the pulse of the city's vibrant energy, inviting you to explore and embrace the marvels of tomorrow.",
"Capture the city of tomorrow, a masterpiece of architectural wonders and technological marvels. The futuristic cityscape rises majestically, with skyscrapers stretching towards the heavens, adorned with vertical gardens that add a touch of nature to the urban landscape. Witness the harmony of humans and advanced AI-driven robots working together to create a thriving metropolis. In the skies above, a fleet of flying cars gracefully glides through the air, providing a mesmerizing spectacle against the backdrop of a vibrant, hyper-realistic sunset. As the city awakens with neon lights and holographic displays, prepare your lens to capture this breathtaking and dynamic urban panorama."
]
opath = "output/future"
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