# image_generator
An image generator tool integrating with an LLM using prompt engineering for the generation logic

### Setup 
Ensure you have python 3.11.9 installed.

* Virtual env setup:
To set up a virtual environment run the following:
```commandline
py -m venv env
env\Scripts\activate
```
That should enable a virtual environment for you to start working with

* Install all necessary requirements by running:
`py -m pip install requirements.txt`

* You will need Nvidia CUDA drivers to run pytorch, you can find
the installation [here](https://pytorch.org/get-started/locally/). Just make sure to choose Pip
and the CUDA version you have, which you can find by running `nvcc --version`.
If everything is installed then run:
`py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

WARNING: If you don't install this way then you will see the following error: `AssertionError: Torch not compiled with CUDA enabled`

* (Optional) Login to hugging face:
For some larger models, i.e. `stable-diffusion-3-medium-diffusers` you need to login to hugging face using the following command:
  `huggingface-cli login`. 
Ensure you have generated an access token in the [hugging face settings](https://huggingface.co/settings/tokens) and when prompted insert the token.
You can set you token as a git credential (type Y).

### RUN
To run execute `py main.py`