from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from composable_diffusion.composable_stable_diffusion.pipeline_composable_stable_diffusion import ComposableStableDiffusionPipeline

import sys
import os
from pathlib import Path
sys.path.append(str(Path('.').resolve().parent))
from code import concept_pj, utils
from code.concept_pj import *
from code.utils import *

def save_images(images, outfolder, prefix):
    """
    Save a list of PIL images to the specified output folder with a given prefix.

    Args:
        images (list): List of PIL images to be saved.
        outfolder (str): Directory where the images should be saved.
        prefix (str): Prefix for the image filenames.

    Returns:
        None
    """
    
    # Ensure the output directory exists
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # Loop through the images and save each one
    for i, image in enumerate(images):
        # Construct the filename
        filename = os.path.join(outfolder, f"{prefix}_{i}.png")
        
        # Save the PIL image
        image.save(filename)


## make name, content, target_style as arguments from argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='mall')
parser.add_argument('--common_content', type=str, default='A sunset at the beach')
parser.add_argument('--content', type=str, default='a 1990s supermarket packed to the brim with people, showcasing a lively, shoulder-to-shoulder shopping experience.')
parser.add_argument('--common_style', type=str, default='photorealistic style')
parser.add_argument('--target_style', type=str, default='renaissance-style painting')
## add argument called new, with action store true
args = parser.parse_args()
name = args.name
common_content = args.common_content
content = args.content
target_style = args.target_style
common_style = args.common_style
print(name)
print(content)
print(target_style)

outfolder = f"fig/{name}"
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

pipe_compose = ComposableStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")  

num_inference_steps = 50
guidance_scale=10
n_im = 10

generator = torch.manual_seed(123)
latents = torch.randn((n_im, 4, 64, 64),generator=generator, dtype = torch.float16)

prompt0 = f"{content} in {common_style}"
prompt1 = f"{common_content} in {target_style}"
prompt_direct = f"{content} in {target_style}"
prompt_compose = f"{content} | {target_style}"
weights = f"{guidance_scale} | {guidance_scale}"

visual_styles = [
    "Art Deco",
    "minimalist",
    "Baroque",
    "Abstract Expressionist",
    "Cubist",
    "Fauvism",
    "Impressionist",
    "Steampunk",
    "Neoclassical",
    "Japanese Ukiyo-e",
    "Surrealism",
    "Memphis Design",
    "Scandinavian",
    "Bauhaus",
    "Pop Art",
    "Art Nouveau",
    "Street Art",
    "American West",
    "Victorian Gothic",
    "Futurism",
    "photorealistic",
    "Mannerist",
    "Flemish",
    "Byzantine",
    "Medieval",
    "Romanesque",
    "Trompe-l'œil",
    "Dutch Golden Age"
]

visual_styles_prefixed = [common_content + " in " + style + " style" for style in visual_styles]
print(visual_styles_prefixed)

ims_compose = []
with torch.cuda.amp.autocast():
    for i in range(n_im):
        latent = latents[i].unsqueeze(0)
        im = pipe_compose(prompt_compose, guidance_scale=guidance_scale, latents=latent,
                    num_inference_steps=num_inference_steps, weights=weights).images
        ims_compose.append(im[0])
ims_direct = pipe([prompt_direct]*n_im, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, latents=latents).images
ims1 = pipe([prompt1]*n_im, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, latents=latents).images
ims0 = pipe([prompt0]*n_im, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, latents=latents).images

save_images(ims_compose, outfolder, "p_compose")
save_images(ims_direct, outfolder, "p_direct")
save_images(ims1, outfolder, "p1")
save_images(ims0, outfolder, "p0")

sample_projected_concept = make_concept_projector(visual_styles_prefixed, pipe)
ims_pj = []
thresholds = [0.5]
for threshold in thresholds:
    for i in range(n_im):
        latent = latents[i].unsqueeze(0)
        im = sample_projected_concept(prompt1, prompt0, latent, 
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                                        threshold=threshold)
        image = Image.fromarray(im[0])
        ims_pj.append(image)
        image.save(f'{outfolder}/proj_{threshold}_{i}.png')


nrow = 4 + len(thresholds)
grid_image = image_grid(ims_direct + ims_compose + ims1 + ims0 + ims_pj, nrow, n_im)
grid_image.save(f"{outfolder}/{name}.png")


