import os
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
from train_vae_latent_512_low_level_no_average import encoder_low_level
from eegdatasets_leaveone_latent_vae_no_average import EEGDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")

image_processor = VaeImageProcessor()

if hasattr(pipe, 'vae'):
    for param in pipe.vae.parameters():
        param.requires_grad = False

vae = pipe.vae.to(device)
vae.requires_grad_(False)
vae.eval()

def reconstruct(features, train=True):
    counts = 200
    for i in range(counts):
        feature = features[i].unsqueeze(0).to(device)
        x_reconstructed = vae.decode(feature).sample
        img_reconstructed = image_processor.postprocess(x_reconstructed, output_type="pil")
        prefix = "train" if train else "test" 
        save_path = f"/home/tom/fsas/eeg_data/generated_images/tmp/{prefix}/{i+1}.png" 
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

if __name__ == "__main__":
    train_img_features = torch.load("/home/tom/fsas/eeg_data/train_image_latent_512.pt", weights_only=True)['image_latent']
    test_img_features = torch.load("/home/tom/fsas/eeg_data/test_image_latent_512.pt", weights_only=True)['image_latent']

    eeg_encoder = encoder_low_level()
    eeg_encoder.load_state_dict(torch.load(f"/home/tom/fsas/eeg_data/models/contrast/encoder_low_level/sub-08/12-18_15-23/200.pth", weights_only=True))
    

    data_path = "/home/tom/fsas/eeg_data/preprocessed_eeg_data"
    train_dataset = EEGDataset(data_path, subjects=["sub-08"], train=True)
    test_dataset = EEGDataset(data_path, subjects=["sub-08"], train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    evaluate(eeg_encoder, train_loader, train=True)
    evaluate(eeg_encoder, test_loader, train=False)
