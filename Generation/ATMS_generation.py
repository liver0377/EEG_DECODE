import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import combinations
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
# from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
from utils import ddp_utils
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from subject_layers.open_clip.model import CLIP, CLIPVisionCfg, CLIPTextCfg, _build_vision_tower
import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW
from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe

train = False
classes = None
pictures= None

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'

config = {
    "data_path": "/home/tom/fsas/eeg_data/preprocessed_eeg_data",
    "project": "train_pos_img_text_rep",
    "entity": "sustech_rethinkingbci",
    "name": "lr=3e-4_img_pos_pro_eeg",
    "lr": 3e-4,
    "epochs": 50,
    "batch_size": 16,
    "logger": True,
    "encoder_type":'ATMS',
}

embed_dim = 1024
visual_config = {
    "image_size": 224,
    "layers": 32,
    "width": 1280,
    "head_width": 80,
    "patch_size": 14
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []
    
    if train:
        text_directory = "/home/tom/fsas/eeg_data/images/training_images"  
    else:
        text_directory = "/home/tom/fsas/eeg_data/images/test_images"

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        new_description = f"{description}"
        texts.append(new_description)

    if train:
        img_directory = "/home/tom/fsas/eeg_data/images/training_images"
    else:
        img_directory ="/home/tom/fsas/eeg_data/images/test_images"
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images

class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]      
        # print("enc_out", enc_out.shape)
        return enc_out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class ATMS(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=25, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out  

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_img_features(visual_encoder, device, preprocessed_image_cache_all, train=True):
    visual_encoder.eval()
    visual_encoder = visual_encoder.to(device)
    save_features = True
    features_list = []  # List to store features    

    image_features_path = f"/home/tom/fsas/eeg_data/image_features_{'train' if train else 'test'}.pt"
    if os.path.exists(image_features_path):
        cached_eeg_features = torch.load(image_features_path, weights_only=True)
        print(f"Loaded cached features from {image_features_path}")
        return cached_eeg_features.cpu()
    
    batch_size = 16 
    with torch.no_grad():
        for i in range(0, preprocessed_image_cache_all.shape[0], batch_size):
            selected_preprocessed_images = preprocessed_image_cache_all[i: i + batch_size].to(device)
            img_features = visual_encoder(selected_preprocessed_images).float() 
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            features_list.append(img_features)

    if save_features:
        features_tensor = torch.cat(features_list, dim=0)
        torch.save(features_tensor.cpu(), image_features_path)  # Save features as .pt file
        print(f"image features saved in {image_features_path}")

    return features_tensor.cpu()

def get_eeg_features(sub, eeg_model, dataloader, device, train=True):
    eeg_model.eval()
    eeg_model = eeg_model.to(device)
    save_features = True
    features_list = []  # List to store features    
    
    eeg_features_path = f"/home/tom/fsas/eeg_data/ATM_S_eeg_features_{sub}_{'train' if train else 'test'}.pt"
    if os.path.exists(eeg_features_path):
        cached_eeg_features = torch.load(eeg_features_path, weights_only=True)
        print(f"Loaded cached features from {eeg_features_path}")
        return cached_eeg_features.cpu()
    
    with torch.no_grad():
        for batch_idx, (eeg_data, _, _, _, _, _, _) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)

            eeg_features = eeg_model(eeg_data, subject_ids)
            features_list.append(eeg_features)


    if save_features:
        features_tensor = torch.cat(features_list, dim=0)
        torch.save(features_tensor.cpu(), eeg_features_path)  # Save features as .pt file
        print(f"eeg features saved in {eeg_features_path}")
    return features_tensor.cpu()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='diffusion prior training')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    ddp_utils.init_distributed_mode(args)
    data_path = config['data_path'] 
    embedding_img_test = torch.load("/home/tom/fsas/eeg_data/ViT-H-14_features_test.pt", weights_only=True)
    embedding_img_train = torch.load("/home/tom/fsas/eeg_data/ViT-H-14_features_train.pt", weights_only=True)

    eeg_encoder = ATMS(63, 250)
    eeg_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ATMS/sub-08/12-13_17-57/40.pth"
    eeg_encoder.load_state_dict(torch.load(eeg_encoder_ckpt_path, weights_only=True))

    visual_encoder = _build_vision_tower(embed_dim, visual_config) 
    visual_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/sub-08/12-13_17-57/40.pth"
    visual_encoder.load_state_dict(torch.load(visual_encoder_ckpt_path, weights_only=True))

    # 1. 使用visual_encoder编码训练集与测试集图片, 使用EEG encoder编码训练集与测试集eeg信号
    sub = "sub-08"   

    train_dataset = EEGDataset(data_path, subjects=[sub], train=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    preprocessed_image_cache_train_all = train_dataset.preprocessed_image_cache
    eeg_train_embedding = get_eeg_features(sub, eeg_encoder, train_loader, device, train=True) # (66160, 1024)
    img_train_embedding = get_img_features(visual_encoder, device, preprocessed_image_cache_train_all, train=True) # (16540, 1024)
    
    test_dataset = EEGDataset(data_path, subjects=[sub], train=False)    
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    preprocessed_image_cache_test_all = test_dataset.preprocessed_image_cache
    eeg_test_embedding = get_eeg_features(sub, eeg_encoder, test_loader, device, train=False) # (200, 1024)
    img_test_embedding = get_img_features(visual_encoder, device, preprocessed_image_cache_test_all, train=False) # (200, 1024) 

    # 2. 使用pipe训练u-net网络
    print(f"img_train_embedding shape: {img_train_embedding.shape}")
    print(f"eeg_train_embedding shape: {eeg_train_embedding.shape}")
    c_embedding = img_train_embedding.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(66160, 1024)  # (66160, 1024)
    h_embedding = eeg_train_embedding # (66160, 1024)
    diffusion_dataset = EmbeddingDataset(c_embedding, h_embedding)
    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=1024, shuffle=True, num_workers=64)

    diffusion_prior =  DiffusionPriorUNet(embed_dim=1024, cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)

    pipe.train(diffusion_dataloader, num_epochs=150, learning_rate=1e-3)

    diffusion_prior_ckpt_path = f"/home/tom/fsas/eeg_data/diffusion_prior/{sub}/diffusion_prior.pt"
    os.makedirs(os.path.dirname(diffusion_prior_ckpt_path), exist_ok=True)
    torch.save(pipe.diffusion_prior.state_dict(), diffusion_prior_ckpt_path)


