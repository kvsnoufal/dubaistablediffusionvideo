# https://github.com/nateraw/stable-diffusion-videos
from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch

prompts= [
   "Transport yourself to a hidden gem in the heart of Dubai's desert, where time seems to stand still. The wide-angle shot captures the enchanting beauty of an old village tucked away amidst the rolling sand dunes. The sun-kissed, weathered walls of the traditional mud-brick houses exude a timeless charm. Palm-frond roofs and wooden doors tell stories of the village's rich heritage. The golden glow of the setting sun casts a warm hue over the scene, evoking a sense of nostalgia. As you explore the dusty streets, the view extends to the vast desert expanse, creating a picturesque sight that blends the old world with the timeless beauty of nature.",
   "Wide-angle aerial cityscape shot of old Dubai in 1950 with vintage architecture, small buildings, color photo, hyper detailed, hd",
   "Wide-angle aerial cityscape shot of downtown Dubai with Burj Khalifa, tallest building in the world, surrounded by buildings. Day time photo ,natural lighting, realistic, hyper realism ",
    "Capture the city of tomorrow, a masterpiece of architectural wonders and technological marvels. The futuristic cityscape rises majestically, with skyscrapers stretching towards the heavens, adorned with vertical gardens that add a touch of nature to the urban landscape. Witness the harmony of humans and advanced AI-driven robots working together to create a thriving metropolis. In the skies above, a fleet of flying cars gracefully glides through the air, providing a mesmerizing spectacle against the backdrop of a vibrant, hyper-realistic sunset. As the city awakens with neon lights and holographic displays, prepare your lens to capture this breathtaking and dynamic urban panorama."
]
seeds = [5934,1125,6012,5779]

pipeline = StableDiffusionWalkPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                revision="fp16",
                                                torch_dtype=torch.float16, 
                                                use_auth_token=True).to('cuda')
video_path = pipeline.walk(
    prompts=prompts,
    seeds=seeds,
    num_interpolation_steps=200,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dubaitimeline',        # Where images/videos will be saved
    name='v1',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=100,     # Number of diffusion steps per image generated. 50 is good default
)