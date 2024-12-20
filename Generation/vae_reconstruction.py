import os
import torch
import re
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from train_vae_latent_512_low_level_no_average import encoder_low_level
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
from torch.utils.data import DataLoader
from custom_pipeline_low_level import Generator4Embeds
from PIL import Image
from utils.ATMS import ATMS
from diffusion_prior import DiffusionPriorUNet, Pipe


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sdxl_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")

image_processor = VaeImageProcessor()

if hasattr(sdxl_pipe, 'vae'):
    for param in sdxl_pipe.vae.parameters():
        param.requires_grad = False

vae = sdxl_pipe.vae.to(device)
vae.requires_grad_(False)
vae.eval()

def reconstruct(features, train=True):
    counts = 200
    for i in range(counts):
        feature = features[i].unsqueeze(0).to(device)
        x_reconstructed = vae.decode(feature).sample
        img_reconstructed = image_processor.postprocess(x_reconstructed, output_type="pil")
        prefix = "train" if train else "test" 
        # save_path = f"/home/tom/fsas/eeg_data/generated_images/tmp/{prefix}/{i+1}.png" 
        save_path = f"/home/tom/fsas/eeg_data/generated_images/demo/{i}.png"
        img_reconstructed[0].save(save_path)
    
def evaluate(eeg_encoder, data_loader, train=True):
    eeg_encoder = eeg_encoder.to(device)
    eeg_encoder.eval()
    
    count = 0
    for batch_idx, (eeg_data, labels, _, _, _, _) in enumerate(data_loader):
        eeg_data = eeg_data.to(device)
        eeg_feature = eeg_encoder(eeg_data)
        x_reconstructed = vae.decode(eeg_feature).sample
        prefix = "train" if train else "test" 
        img_reconstructed = image_processor.postprocess(x_reconstructed, output_type="pil")
        for i, (label) in enumerate(labels):
            save_path = f"/home/tom/fsas/eeg_data/generated_images/tmp/{prefix}/{label+1}.png" 
            # 每个label生成一张图即可
            if os.path.exists(save_path):
                pass

            img_reconstructed[i].save(save_path)
            count = count + 1 
            if count == 200:
                return

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def generate_10_images_with_low_level(eeg_encoder, eeg_encoder_low_level, sub, data_loader, pipe):
    """
    生成10个label图片
    data_loader应为训练集loader
    """
    eeg_encoder = eeg_encoder.to(device)
    eeg_encoder_low_level = eeg_encoder_low_level.to(device)

    
    seed_value = 42
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_value)
    
    count = 0
    for batch_idx, (eeg_datas, labels, _, _, images, _) in enumerate(data_loader):
        eeg_datas = eeg_datas.to(device)

        batch_size = eeg_datas.size(0)  # Assume the first element is the data tensor
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        eeg_embedding = eeg_encoder(eeg_datas, subject_ids)

        eeg_latent = eeg_encoder_low_level(eeg_datas)
        x_reconstructed = vae.decode(eeg_latent).sample
        img_reconstructed = image_processor.postprocess(x_reconstructed, output_type="pil") # low level image

        generator = Generator4Embeds(num_inference_steps=4, device=device, low_level_image=img_reconstructed)   
        h = pipe.generate(c_embeds=eeg_embedding, num_inference_steps=10, guidance_scale=2.0)
                
        for i, (label) in enumerate(labels):
            save_prefix = f"/home/tom/fsas/eeg_data/generated_images/demo/output"
            final_image_name = f"final_{label}.png" 
            final_image_save_path = os.path.join(save_prefix, final_image_name)

            if os.path.exists(final_image_save_path):
                pass

            final_image = generator.generate(h[i], generator=gen)
            final_image.save(final_image_save_path)

            original_image = images[i]
            image = Image.open(original_image)
            original_image_name = f"original_{label}.png"
            original_image_save_path = os.path.join(save_prefix, original_image_name)
            image.save(original_image_save_path)

            count = count + 1
            if count == 10:
               return 

            
if __name__ == "__main__":
    # train_img_features = torch.load("/home/tom/fsas/eeg_data/train_image_latent_512.pt", weights_only=True)['image_latent']
    # test_img_features = torch.load("/home/tom/fsas/eeg_data/test_image_latent_512.pt", weights_only=True)['image_latent']
    
    # eeg_encoder = encoder_low_level()
    # eeg_encoder.load_state_dict(torch.load(f"/home/tom/fsas/eeg_data/models/contrast/encoder_low_level/sub-08/12-18_15-23/200.pth", weights_only=True))
    

    data_path = "/home/tom/fsas/eeg_data/preprocessed_eeg_data"
    train_dataset = EEGDataset(data_path, subjects=["sub-08"], train=True)
    test_dataset = EEGDataset(data_path, subjects=["sub-08"], train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # evaluate(eeg_encoder, train_loader, train=True)
    # evaluate(eeg_encoder, test_loader, train=False)
    # reconstruct(test_img_features)

    eeg_encoder = ATMS(63, 250)
    eeg_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ATMS/sub-08/12-13_17-57/40.pth"
    eeg_encoder.load_state_dict(torch.load(eeg_encoder_ckpt_path, weights_only=True))

    eeg_encoder_low_level = encoder_low_level() 
    eeg_encoder_low_level_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/encoder_low_level/sub-08/12-18_15-23/200.pth"
    eeg_encoder_low_level.load_state_dict(torch.load(eeg_encoder_low_level_ckpt_path, weights_only=True))

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    diffusion_prior_pipe = Pipe(diffusion_prior, device=device) 
    diffusion_prior_pipe.diffusion_prior.load_state_dict(torch.load(f'/home/tom/fsas/eeg_data/diffusion_prior_old/sub-08/diffusion_prior.pt', map_location=device))

    print(f"generating...")
    generate_10_images_with_low_level(eeg_encoder, eeg_encoder_low_level, "sub-08", train_loader, diffusion_prior_pipe)
