import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Replace the model version with your required version if needed
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)

# Running the inference on GPU with cuda enabled
pipeline = pipeline.to('cuda')

prompt = ("Generate a realistic family friendly image for an event with the following description: "
          "`KidsArt! teaches children art and art appreciation - term time weekly classes,"
          " holiday camps, parties, private classes and art scholarship preparation. "
          "We carry reviews of London art exhibitions and run the KidsArt! Store for"
          " art supplies and children's books.Classes are held at KidsArt!"
          " Children are taught how to paint, draw, print, work with clay, sculpt, collage, "
          "use soft and oil pastels, and other core techniques, whilst at the same time learning "
          "about famous artists and their styles - each project is inspired by an artist or art movement."
          "that accepts kids between 18 months - 3 years and the event title "
          "is `Lisa Gilbert Academy of Ballet and Performing Arts - Baby Ballet` ")


image = pipeline(prompt=prompt).images[0]

if __name__ == "__main__":
    # torch.cuda.is_available()
    image.show()
