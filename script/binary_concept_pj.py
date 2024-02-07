# !nvidia-smi
# !pip install diffusers==0.8.0
# !pip install transformers scipy ftfy
# !pip install "ipywidgets>=7,<8"

import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
parser.add_argument('--n_im', type=int, default=5, help='Number of images to generate')
parser.add_argument('--prompt_plus', type=str, required=True, help='Prompt for positive direction')
parser.add_argument('--prompt_minus', type=str, required=True, help='Prompt for negative direction')
parser.add_argument('--prompt_z', type=str, required=True, help='Prompt for target Z')
parser.add_argument('--prompt0', type=str, required=True, help='Original prompt')


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  
pipe = pipe.to("cuda")
tokenizer, text_encoder, unet, scheduler, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.scheduler, pipe.vae

max_length = tokenizer.model_max_length
torch_device = 'cuda'
height=512
width=512

def negative_prompt(prompt, neg_prompt, latents, num_inference_steps=50, guidance_scale=15): 
    batch_size = latents.size()[0]
    neg_input = tokenizer(
      neg_prompt * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    neg_embeddings = text_encoder(neg_input.input_ids.to(torch_device))[0]
    text_input_tmp = tokenizer(prompt * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    emb0 = text_encoder(text_input_tmp.input_ids.to(torch_device))[0]
    latents = latents.to(torch_device) 
    scheduler.set_timesteps(num_inference_steps)
    
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, 
                                  encoder_hidden_states=torch.cat([neg_embeddings, emb0])).sample
                # perform guidance
                noise_pred_neg, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_neg + guidance_scale * (noise_pred_text - noise_pred_neg)
                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    return images


def make_concept_projector(prompt_plus, prompt_minus):
    """
    Implement "concept projection" and return a function `_concept_proj`
   
    Args:
        prompt_plus, prompt_minus: they are as defined in this paper 
                (Algorithm 2, corresponding to \gamma_{+}, \gamma_{-})
    """
    
    def _concept_proj(prompt, prompt_z, latents, num_inference_steps=50, guidance_scale=15): 
        """
        Returns an image
        
        Args:
            prompt: it's as defined in this paper 
                    (Algorithm 2, corresponding to \gamma_{1})
            prompt_z: ...
            latents: Pre-generated noisy latents, sampled from a Gaussian distribution, 
                    to be used as inputs for image generation.
            num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
            guidance_scale: Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). 
                    `guidance_scale` is defined as `w` of equation 2
        """
        batch_size = latents.size()[0]
        uncond_input = tokenizer(
          [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_input_tmp = tokenizer(prompt * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        emb0 = text_encoder(text_input_tmp.input_ids.to(torch_device))[0]
        text_input_tmp = tokenizer(prompt_plus * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        emb_z0 = text_encoder(text_input_tmp.input_ids.to(torch_device))[0]
        text_input_tmp = tokenizer(prompt_minus * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        emb_z1 = text_encoder(text_input_tmp.input_ids.to(torch_device))[0]
        text_input_tmp = tokenizer(prompt_z * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        emb_z_target = text_encoder(text_input_tmp.input_ids.to(torch_device))[0]

        # latents = latents * scheduler.init_noise_sigma
        latents = latents.to(torch_device) 
        scheduler.set_timesteps(num_inference_steps)

        with autocast("cuda"):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 5)
                    noise_pred = unet(latent_model_input, t, 
                                      encoder_hidden_states=torch.cat([uncond_embeddings, emb0, emb_z0, emb_z1, emb_z_target])).sample                
                    noise_pred_uncond, noise_pred_text0, noise_pred_text_z0, noise_pred_text_z1, noise_pred_text_z_target = noise_pred.chunk(5)

                    ## score difference
                    noise_tmp = noise_pred_text0 - noise_pred_text_z_target                    
                    ## Z direction
                    u = noise_pred_text_z1 - noise_pred_text_z0
                    u /= torch.sqrt((u**2).sum())
                    ## project out Z direction
                    noise_pred_text0 -= (noise_tmp * u).sum() * u

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text0 - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images
    
    return _concept_proj

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# Parse the arguments
args = parser.parse_args()
# Set the seed
seed = args.seed
n_im = args.n_im
prompt_plus = args.prompt_plus
prompt_minus = args.prompt_minus
prompt_z = args.prompt_z
prompt0 = args.prompt0

generator = torch.manual_seed(seed)
batch_size = 1
num_inference_steps = 50
guidance_scale = 15
latents = torch.randn((n_im, 4, 64, 64),generator=generator, dtype = unet.dtype)
sample_projected_concept = make_concept_projector(prompt_plus, prompt_minus)


img_folder = (prompt0 + " " + prompt_z).replace(" ", "_")
img_folder = f"../figures/fig_pj_binary/" + img_folder
if not os.path.exists(img_folder):
    os.makedirs(img_folder)


## direct prompting
method_name = "direct_prompt"
ims = []
for i in range(n_im):
    latent = latents[i].unsqueeze(0)
    im_ = negative_prompt(prompt0, "",latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    im_ = Image.fromarray(im_[0])
    ims.append(im_)
grid_image = image_grid(ims, 1, n_im)
grid_image.save(f"{img_folder}/{method_name}.png")


method_name = "concept_proj"
ims = []
for i in range(n_im):
    latent = latents[i].unsqueeze(0)
    im_ = sample_projected_concept(prompt0, prompt_z,latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    im_ = Image.fromarray(im_[0])
    ims.append(im_)
grid_image = image_grid(ims, 1, n_im)
grid_image.save(f"{img_folder}/{method_name}.png")



