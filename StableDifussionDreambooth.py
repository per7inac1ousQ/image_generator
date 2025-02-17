from diffusers import DiffusionPipeline, AutoencoderKL
import torch

from main import model_id

PROJECT_NAME = "Dreambooth_SDXL"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR = "/kaggle/input/abids-photos"
REPO_ID = "kingabzpro/sdxl-lora-abid"


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    model_id=MODEL_NAME,
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
