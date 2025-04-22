import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import re
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from train_vae_latent_512_low_level_no_average import encoder_low_level
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
from torch.utils.data import DataLoader
from custom_pipeline_low_level import Generator4Embeds
from PIL import Image
from utils.reconstruction_utils import ATMS, CLIPEncoder
from diffusion_prior import DiffusionPriorUNet, Pipe


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sdxl_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")

image_processor = VaeImageProcessor()
clip_encoder = CLIPEncoder().to(device)

if hasattr(sdxl_pipe, 'vae'):
    for param in sdxl_pipe.vae.parameters():
        param.requires_grad = False

vae = sdxl_pipe.vae.to(device)
vae.requires_grad_(False)
vae.eval()

def generate_low_level(eeg_encoder_low_level, eeg_datas, labels):
    """
    labels与eeg_datas的样本数量一致
    """
    assert eeg_datas.shape[0] == labels.shape[0], "eeg_datas and labels do not match"
    
    os.makedirs("/home/tom/fsas/eeg_data/generated_images/low_level", exist_ok=True)
    eeg_encoder_low_level = eeg_encoder_low_level.to(device)
    eeg_datas = eeg_datas.to(device)

    with torch.no_grad():
        for i, (label) in enumerate(labels):
            eeg_latent = eeg_encoder_low_level(eeg_datas[i].unsqueeze(0)) 
            x_reconstructed = vae.decode(eeg_latent).sample
            img_reconstructed = image_processor.postprocess(x_reconstructed, output_type="pil")  
            save_path = f"/home/tom/fsas/eeg_data/generated_images/low_level/{label}.png"
            img_reconstructed[0].save(save_path)
    
def clip_low_level_pipeline(low_level_image_path, eeg_encoder, eeg_datas, labels, sub, pipe):
    """
    eeg_datas与labels样本量必须一致
    """

    eeg_encoder = eeg_encoder.to(device)
    eeg_datas = eeg_datas.to(device)
    labels = labels.to(device)

    assert eeg_datas.shape[0] == labels.shape[0], "eeg_datas and labels do not match"
    
    os.makedirs("/home/tom/fsas/eeg_data/generated_images/clip_low-level", exist_ok=True)

    seed_value = 42
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_value) 

    for i, (label) in enumerate(labels):
        low_level_image = Image.open(low_level_image_path + '/' + f"{label}.png")
        
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((1,), subject_id, dtype=torch.long).to(device)

        eeg_data = eeg_datas[i].unsqueeze(0)
        print(f"eeg_data shape: {eeg_data.shape}")
        eeg_features = eeg_encoder(eeg_data, subject_ids)   # [1, 1024]
        low_level_image = clip_encoder.preprocess(low_level_image, return_tensors="pt").pixel_values # [1, 3, 512, 512]
        
        generator = Generator4Embeds(num_inference_steps=4, device=device, low_level_image=low_level_image) 
        h = pipe.generate(c_embeds=eeg_features, num_inference_steps=10, guidance_scale=2.0)
        generated_image = generator.generate(h, generator=gen)

        
        clip_low_level_path = f"/home/tom/fsas/eeg_data/generated_images/clip_low-level/{label}.png"
        generated_image.save(clip_low_level_path)
        

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

if __name__ == "__main__":
    data_path = "/home/tom/fsas/eeg_data/preprocessed_eeg_data"
    test_dataset = EEGDataset(data_path, subjects=["sub-08"], train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    eeg_datas = test_dataset.data
    labels = test_dataset.labels

    print(f"eeg_datas shape: {eeg_datas.shape}")
    print(f"labels shape: {labels.shape}")
 

    eeg_encoder = ATMS(63, 250)
    eeg_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ATMS/sub-08/12-13_17-57/40.pth"
    eeg_encoder.load_state_dict(torch.load(eeg_encoder_ckpt_path, weights_only=True))

    eeg_encoder_low_level = encoder_low_level() 
    eeg_encoder_low_level_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/encoder_low_level/sub-08/12-18_15-23/200.pth"
    eeg_encoder_low_level.load_state_dict(torch.load(eeg_encoder_low_level_ckpt_path, weights_only=True))

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    diffusion_prior_pipe = Pipe(diffusion_prior, device=device) 
    diffusion_prior_pipe.diffusion_prior.load_state_dict(torch.load(f'/home/tom/fsas/eeg_data/diffusion_prior_old/sub-08/diffusion_prior.pt', map_location=device))

    low_level_image_path = "/home/tom/fsas/eeg_data/generated_images/low_level"
    clip_low_level_pipeline(low_level_image_path, eeg_encoder, eeg_datas, labels, "sub-08", diffusion_prior_pipe)
