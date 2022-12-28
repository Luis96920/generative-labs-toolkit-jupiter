import argparse, os, sys, glob
from crypt import methods
from fileinput import filename
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from utils import upload_gcs_file

import firebase_admin
from firebase_admin import credentials, firestore

from flask import Flask, request
import random

app = Flask(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

# DB CONFIG
cred = credentials.Certificate("firestore_token.json")
firebase_admin.initialize_app(cred)
db = firestore.client()




# STATIC CONFIG
precision = "autocast"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
config = "configs/stable-diffusion/v1-inference.yaml"
outdir = "outputs/txt2img-samples"
config = OmegaConf.load(config)
model = load_model_from_config(config, ckpt)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
sampler = PLMSSampler(model)


@app.route("/", methods=["POST"])
def img2img():
    ### CODIGO IMG2img
    pass

@app.route("/", methods=["POST"])
def inpainting():
    ### CODIGO inpainting
    pass

# SELECTIONS
@app.route('/', methods=['POST'])
def main():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        payload = request.json
        job_id = payload.get('job_id')
        prompt = str(payload.get("prompt", 512))
        H = int(payload.get("H", 512))
        W = int(payload.get("W", 512))
        f = int(payload.get("F", 8))
        C = int(payload.get("C", 4))
        n_iter = int(payload.get("n_iter", 1))
        scale = float(payload.get("scale", 7.5))
        n_samples = int(payload.get("n_samples", 1))
        ddim_eta = float(payload.get("ddim_eta", 0.0))
        ddim_steps = int(payload.get("ddim_steps", 50))
        skip_grid = payload.get("skip_grid", True)

        batch_size = n_samples
        n_rows = batch_size

        prompt = prompt
        data = [batch_size * [prompt]]

        os.makedirs(outdir, exist_ok=True)
        outpath = outdir
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)



        seed = random.randint(1,200000)
        seed_everything(seed)

        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None

        precision_scope = autocast if precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    urls_to_images = []
                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):

                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                                conditioning=c,
                                                                batch_size=n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=scale,
                                                                unconditional_conditioning=uc,
                                                                eta=ddim_eta,
                                                                x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)


                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                filename = f"{job_id}-{base_count}.png"
                                destination = os.path.join(sample_path, filename)
                                img.save(destination)
                                upload_gcs_file(destination, destination)
                                base_count += 1

                                
                                urls_to_images.append(filename)
                                

                            if not skip_grid:
                                all_samples.append(x_checked_image_torch)

                   
                    if not skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        grid_filename = f'{job_id}-grid.png'
                        destination = os.path.join(outpath, grid_filename)
                        img.save(destination)
                        upload_gcs_file(destination, destination)
                        
                        
                        db.collection("job_collection").document(job_id).update({"url_to_image_grid": grid_filename})

                    db.collection("job_collection").document(job_id).set({"params" : {"seed": seed}, "urls_to_images": urls_to_images}, merge=True)
                    os.system(f'rm -rf {sample_path}/*.png')
                    os.system(f'rm -rf {outpath}/*.png')
                    torch.cuda.empty_cache()


        return grid_filename if not skip_grid else urls_to_images[0]
    else:
        return 'Content-Type not supported!'

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))


