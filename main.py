import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

# Replace the model version with your required version if needed
pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)

# Running the inference on GPU with cuda enabled
pipeline = pipeline.to('cuda')

prompt = ("a slender punk female sitting reclined against the wall of a dark street savoring a short cigarette, on the wall spray painted in red "
          "\"Punk\'s Not Dead\", hyper realistic, detailed dreads pink and black hair, side mohawk haircut, large necklaces, tattoos on all body, nose lip eyebrow piercing, worn crop tank top, dirty and torn apart cloths, wide dark background, luis royo illustration, rembrandt lighting, concept art, fantasy art, hyper detailed, intricate, sharp focus, best quality, masterpiece")

image = pipeline(
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

if __name__ == "__main__":
    # torch.cuda.is_available()
    image.show()
