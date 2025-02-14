import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

torch.set_float32_matmul_precision("high")

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"


pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None
)
pipeline.set_progress_bar_config(disable=True)

# Running the inference on GPU with cuda enabled
prompt = ("slender punk female sitting reclined against the wall of a dark street savoring a short cigarette,"
          "hyper realistic, detailed dreads pink and black hair, side mohawk haircut, large necklaces, tattoos on all body, nose lip eyebrow piercing, worn crop tank top, dirty and torn apart cloths, wide dark background, luis royo illustration, rembrandt lighting, concept art, fantasy art, hyper detailed, intricate, sharp focus, best quality, masterpiece")

image = pipeline(
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=28,
    max_sequence_length=128,
    guidance_scale=7.0,
).images[0]
pipeline = pipeline.to('cuda')


if __name__ == "__main__":
    # torch.cuda.is_available()
    image.show()
