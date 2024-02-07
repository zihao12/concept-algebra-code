## methods for concept algebra
import torch
from tqdm.auto import tqdm
from torch import autocast

def get_text_embeddings(prompts, pipeline):
    text_input = pipeline.tokenizer(prompts, padding="max_length", 
                            max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    
    return text_embeddings

def select_K(cumulative_proportions_of_variance_explained, threshold):
    for i, proportion in enumerate(cumulative_proportions_of_variance_explained):
        if proportion > threshold:
            K = i + 1
            break
    return max(K, 1)

def make_concept_projector(prompts_z, pipe):
    def _concept_proj(prompt_target, prompt_original, latents, num_inference_steps=50, guidance_scale=10, gamma = 1, threshold = 0.9, strength = 1): 
        
        batch_size = latents.size()[0]
        emb_uncond = get_text_embeddings([""] * batch_size, pipe)
        
        emb_new = get_text_embeddings([prompt_target] * batch_size, pipe)
        emb_original = get_text_embeddings([prompt_original] * batch_size, pipe)
        
        embs_z = get_text_embeddings(prompts_z, pipe)
        
        # latents = latents * scheduler.init_noise_sigma
        latents = latents.to(pipe.device) 
        pipe.scheduler.set_timesteps(num_inference_steps)
        t1 = 1000 * strength

        with autocast("cuda"):
            for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):
                with torch.no_grad():
                    if t > t1:
                        h = torch.cat([emb_uncond, emb_original])
                        noise_pred = pipe.unet(torch.cat([latents] * 2), t, encoder_hidden_states=h).sample 
                        noise_pred_uncond, noise_pred_text0 = noise_pred.chunk(2)
                    else:
                        ## compute projection matrix
                        n = embs_z.size()[0]
                        _, p1, p2, p3 = latents.size()
                        d = p1*p2*p3
                        A = pipe.unet(torch.cat([latents] * n), ## n * 4 * 64 * 64
                                                t, encoder_hidden_states= embs_z).sample
                        A = A / torch.sqrt(torch.tensor(float(n)))
                        A = A.permute(1, 2, 3, 0).reshape(d, n)
                        A = A - A.mean(dim=1, keepdim=True)
                        # Q, _ = torch.linalg.qr(A.to(torch.float32)) 
                        # Q = Q.to(torch.float16)
                        A = A.to(torch.float32)
                        Q, S, _ = torch.linalg.svd(A, full_matrices=False)
                        Q = Q.to(torch.float16)
                        # print("Q:", Q.size())
                        variance_explained = S ** 2
                        total_variance = variance_explained.sum()
                        proportions_of_variance_explained = variance_explained / total_variance
                        cumulative_proportions_of_variance_explained = proportions_of_variance_explained.cumsum(dim = 0)

                        # print("Cumulative proportions of variance explained:", cumulative_proportions_of_variance_explained)
                        K = select_K(cumulative_proportions_of_variance_explained, threshold)
                        Q = Q[:, :K]

                        ## compute the two scores
                        h = torch.cat([emb_uncond, emb_original, emb_new])
                        noise_pred = pipe.unet(torch.cat([latents] * 3), t, encoder_hidden_states=h).sample   
                        noise_pred_uncond, noise_pred_text0, noise_pred_text1 = noise_pred.chunk(3)
                        
                        diff = noise_pred_text1 - noise_pred_text0
                        diff = diff.permute(1, 2, 3, 0).reshape(d, 1)
                        diff = Q @ (Q.T @ diff)
                        diff = diff.reshape(p1, p2, p3, 1).permute(3, 0, 1, 2)
                        noise_pred_text0 += gamma * diff
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text0 - noise_pred_uncond)
                    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images
    
    return _concept_proj    


def make_concept_transfer(prompt_plus, prompt_minus,pipe):
    def _concept_transfer(prompt, latents, num_inference_steps=50, guidance_scale=10, gamma = 1): 
        
        batch_size = latents.size()[0]
        emb_uncond = get_text_embeddings([""] * batch_size, pipe)
        emb_original = get_text_embeddings([prompt] * batch_size, pipe)        
        emb_plus = get_text_embeddings([prompt_plus] * batch_size, pipe)
        emb_minus  = get_text_embeddings([prompt_minus] * batch_size, pipe)
        
        # latents = latents * scheduler.init_noise_sigma
        latents = latents.to(pipe.device) 
        pipe.scheduler.set_timesteps(num_inference_steps)

        with autocast("cuda"):
            for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):
                with torch.no_grad():
                    ## compute projection matrix
                    h = torch.cat([emb_uncond, emb_original, emb_plus, emb_minus])
                    noise_pred = pipe.unet(torch.cat([latents] * 4), t, encoder_hidden_states=h).sample   
                    noise_pred_uncond, noise_pred_text0, noise_pred_plus, noise_pred_minus = noise_pred.chunk(4)
                    
                    noise_pred_text0 = noise_pred_text0 + noise_pred_plus - noise_pred_minus
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text0 - noise_pred_uncond)
                    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = pipe.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images
    
    return _concept_transfer