from diffusers import DiffusionPipeline, AutoencoderKL
import torch

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda");


prompt = ("a photorealistic picture of 3 kids with perfect symmetrical eyes at a drawing class painting and smiling without their mouth open")
image = pipe(prompt=prompt, num_inference_steps=150, num_images_per_prompt = 1).images[0]


if __name__ == "__main__":
    # torch.cuda.is_available()
    image.show()
