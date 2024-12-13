import os

import torch
import time
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import time

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset, vlmodel, preprocess_train

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
from PIL import Image
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
from utils import ddp_utils
import torch.backends.cudnn as cudnn


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
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
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

def train_model(sub, eeg_model, image_model, dataloader, optimizer, device, img_features_all, preprocessed_image_cache_train_all,  config, scaler):
    eeg_model.train()
    image_model.train()

    # img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.90
    mse_loss_fn = nn.MSELoss()
    print("epoch begin")
    for batch_idx, (eeg_data, labels, _, _, indices, _, img_features) in enumerate(dataloader):
        if batch_idx % 100 == 0:
            print(f"batch: {batch_idx + 1}") 
        

        batch_size = eeg_data.shape[0]
        eeg_data = eeg_data.to(device)
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        
        
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
  
        
        # 获取eeg特征
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # start_time = time.time()
            # print(f"getting eeg feature")
            eeg_features = eeg_model.module(eeg_data, subject_ids).float()        
            # features_list.append(eeg_features)
        
            # eeg_end_time = time.time()
            # print(f"eeg 编码时间: {eeg_end_time - start_time} seconds")
            # 获取图片特征
            # image_inputs = torch.stack([preprocess_train(Image.open(img_path).convert("RGB")) for img_path in img])
            # print(f"img length: {len(img)}")
            # print(f"getting image feature")
            selected_preprocessed_images = preprocessed_image_cache_train_all[indices].to(device)
            img_features_model = image_model.module(selected_preprocessed_images).float()
            img_features_model = img_features_model / img_features_model.norm(dim=-1, keepdim=True)
            # print(f"img_features_model shape: {img_features_model.shape}")

            # image_end_time = time.time()
            # print(f"图片编码时间: {image_end_time - eeg_end_time} seconds")
            # img_features_model = img_features_model / img_features_model.norm(dim=-1, keepdim=True)
        
            # 更新全局图片embedding
            # print(f"updating global image embedding")
            # if (batch_idx +  1) % 10 == 0:
            # img_features_all[indices] = img_features_model.detach().float()#.cpu()

            # update_end_time = time.time()

            # if (batch_idx + 1) % 10 == 0:
            # print(f"特征更新时间: {update_end_time - image_end_time} seconds")
        
        
            # 计算loss
            # print(f"caculating loss")
            logit_scale = eeg_model.module.logit_scale
            img_loss = eeg_model.module.loss_func(eeg_features, img_features_model, logit_scale)
            regress_loss =  mse_loss_fn(eeg_features, img_features_model)
            loss = (alpha * regress_loss *10 + (1 - alpha) * img_loss*10)

            # loss_end_time = time.time()
            # print(f"loss 计算时间: {loss_end_time - update_end_time} seconds")
            # print("backward")

        # loss.backward()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        # backward_end_time = time.time()
        # print(f"反向传播时间: {backward_end_time - update_end_time} seconds")
 
        # 计算精确率
        # img_features_1654 = []
        # for i in range(1654):
        #     img_features = image_model.module(preprocessed_image_cache_train_all[i * 10]).float()
        #     img_features_1654.append(img_features)
        # img_features_1654 = torch.cat(img_features_1654, dim=-1)
        # # img_features_1654 = img_features_all[::10].to(device).float()
        # logits_img = logit_scale * eeg_features @ img_features_1654.T
        # logits_single = logits_img
        # predicted = torch.argmax(logits_single, dim=1)
        
        total += batch_size
        # correct += (predicted == labels).sum().item()

    
    average_loss = total_loss / (batch_idx+1)
    # accuracy = correct / total

    return average_loss# , accuracy# , torch.cat(features_list, dim=0)

def evaluate_model(sub, eeg_model, image_model, dataloader, device, img_features_all, k,  preprocessed_image_cache_test_all, config):
    eeg_model.eval()
    image_model.eval()

    # img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    all_labels = set(range(img_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _, _, indices, _, _) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            # img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            eeg_features = eeg_model.module(eeg_data, subject_ids)
            
            selected_preprocessed_images = preprocessed_image_cache_test_all[indices].to(device)
            img_features_model = image_model.module(selected_preprocessed_images).float() 
            img_features_model = img_features_model / img_features_model.norm(dim=-1, keepdim=True)
        
            logit_scale = eeg_model.module.logit_scale 
            img_loss = eeg_model.module.loss_func(eeg_features, img_features_model, logit_scale)
            regress_loss =  mse_loss_fn(eeg_features, img_features_model)
            loss = (alpha * regress_loss *10 + (1 - alpha) * img_loss*10)
                
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes].to(device).float()
                
                if k==200:
                    # Compute corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    
                    # logits_single is the model output, assumed to be shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get top-5 predicted indices
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    # Check if true label is in top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    # Check if true label is in top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            # del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, current_time, eeg_model, image_model, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, preprocessed_image_cache_train_all, preprocessed_image_cache_test_all, config, logger=None, scaler=None):

    if ddp_utils.is_main_process():
        logger = wandb_logger(config) if logger else None
        logger.watch(eeg_model,logger) 

    # train_losses, train_accuracies = [], []
    train_losses = []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    # best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(config.epochs):
        # Train the model
        train_dataloader.sampler.set_epoch(epoch)

        
        if not config.evaluate:
            train_loss  = train_model(sub, eeg_model, image_model, train_dataloader, optimizer, device,  img_features_train_all, preprocessed_image_cache_train_all,config=config, scaler=scaler)
            if ddp_utils.is_main_process():
                if (epoch +1) % 5 == 0:                    
                    # Save the model every 5 epochs                  
                    if config.insubject==True:       
                        os.makedirs(f"/home/tom/fsas/eeg_data/models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)
                        os.makedirs(f"/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/{sub}/{current_time}", exist_ok=True)
                        eeg_encoder_file_path = f"/home/tom/fsas/eeg_data/models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                        image_encoder_file_path = f"/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/{sub}/{current_time}/{epoch+1}.pth"
                        # os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
                        # os.makedirs(f"./models/contrast/ImageEncoder/{sub}/{current_time}", exist_ok=True)
                        # eeg_file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                        # image_encoder_file_path = f"./models/contrast/ImageEncoder/{sub}/{current_time}/{epoch+1}.pth"
                        torch.save(eeg_model.module.state_dict(), eeg_encoder_file_path)            
                        torch.save(image_model.module.state_dict(), image_encoder_file_path)
                    else:                
                        os.makedirs(f"/home/tom/fsas/eeg_data/models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)
                        os.makedirs(f"/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/{sub}/{current_time}", exist_ok=True)
                        eeg_encoder_file_path = f"/home/tom/fsas/eeg_data/models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                        image_encoder_file_path = f"/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/{sub}/{current_time}/{epoch+1}.pth"
                        # os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
                        # os.makedirs(f"./models/contrast/ImageEncoder/{sub}/{current_time}", exist_ok=True)
                        # eeg_file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                        # image_encoder_file_path = f"./models/contrast/ImageEncoder/{sub}/{current_time}/{epoch+1}.pth"
                        torch.save(eeg_model.module.state_dict(), eeg_encoder_file_path)            
                        torch.save(image_model.module.state_dict(), image_encoder_file_path) 
                    print(f"EEG Encoder Model saved in {eeg_encoder_file_path}!")
                    print(f"Visual Encoder Model saved in {image_encoder_file_path}")
                train_losses.append(train_loss)
            # train_accuracies.append(train_accuracy)


            # Evaluate the model
            # 每一个epoch都要重新对img_features_test_all进行计算
        if ddp_utils.is_main_process():
            with torch.no_grad():
                batch_size = 1 
                num_samples = preprocessed_image_cache_test_all.size(0)
                img_features_test_all = []
                for i in range(0, num_samples, batch_size):
                    batch = preprocessed_image_cache_test_all[i:i+batch_size].to(device)
                    img_embedding = image_model.module(batch).cpu().float()
                    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    img_features_test_all.append(img_embedding)

                # 拼接所有批次结果
                img_features_test_all = torch.cat(img_features_test_all, dim=0)  

            # preprocessed_image_cache_test_all = preprocessed_image_cache_test_all.to(device)
            # for img in preprocessed_image_cache_test_all:
            #     img.to(device)
            #     img_embedding = image_model.module(img) 
            #     img_features_test_all.append(img_embedding)
            # img_features_test_all = image_model.module(preprocessed_image_cache_test_all).float()
            # img_features_test_all = img_features_test_all / img_features_test_all.norm(dim=-1, keepdim=True)
            # img_features_test_all = torch.cat(img_features_test_all, dim=0)
                
            test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, image_model, test_dataloader, device,  img_features_test_all,k=200, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            _, v2_acc, _ = evaluate_model(sub, eeg_model, image_model, test_dataloader, device, img_features_test_all, k = 2, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            _, v4_acc, _ = evaluate_model(sub, eeg_model, image_model, test_dataloader, device, img_features_test_all, k = 4, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            _, v10_acc, _ = evaluate_model(sub, eeg_model, image_model, test_dataloader, device, img_features_test_all, k = 10, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, image_model, test_dataloader, device, img_features_test_all,  k=50, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, image_model, test_dataloader, device, img_features_test_all,  k=100, preprocessed_image_cache_test_all=preprocessed_image_cache_test_all, config=config)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            v2_accs.append(v2_acc)
            v4_accs.append(v4_acc)
            v10_accs.append(v10_acc)
        
            # Append results for this epoch
            epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            # "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc":top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc":v50_top5_acc,
            "v100_top5_acc": v100_top5_acc
            }

            results.append(epoch_results)
            # If the test accuracy of the current epoch is the best, save the model and related information
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                # best_model_weights = model.state_dict().copy()
            
                best_epoch_info = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    # "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "v2_acc":v2_acc,
                    "v4_acc":v4_acc,
                    "v10_acc":v10_acc
                }
            logger.log({
               "Train Loss": train_loss,
               #  "Train Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy,
                "v2 Accuracy": v2_acc,
                "v4 Accuracy": v4_acc,
                "v10 Accuracy": v10_acc,
                "Epoch": epoch
            })

            print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
            # print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
            print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
        # # Load best model weights
        # model.load_state_dict(best_model_weights)

        # # # Save best model
        # torch.save(model.state_dict(), '{train_pos_img_text}.pth')
        # Create 5 subplots
        dist.barrier()
        torch.cuda.empty_cache()
    if ddp_utils.is_main_process():
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        # Loss plot
        axs[0, 0].plot(train_losses, label='Train Loss')
        axs[0, 0].plot(test_losses, label='Test Loss')
        axs[0, 0].legend()
        axs[0, 0].set_title("Loss Curve")

        # Overall accuracy plot
        # axs[0, 1].plot(train_accuracies, label='Train Accuracy')
        axs[0, 1].plot(test_accuracies, label='Test Accuracy')
        axs[0, 1].legend()
        axs[0, 1].set_title("Accuracy Curve")

        # The following are the three new plots you added, assuming you have calculated the corresponding accuracies
        # 2-class accuracy plot
        axs[1, 0].plot(v2_accs, label='2-class Accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_title("2-Class Accuracy Curve")

        # 4-class accuracy plot
        axs[1, 1].plot(v4_accs, label='4-class Accuracy')
        axs[1, 1].legend()
        axs[1, 1].set_title("4-Class Accuracy Curve")

        # 10-class accuracy plot
        axs[2, 0].plot(v10_accs, label='10-class Accuracy')
        axs[2, 0].legend()
        axs[2, 0].set_title("10-Class Accuracy Curve")

        # Construct the string information you want to annotate
        info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                    f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                    # f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                    f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                    f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                    f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                    f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                    f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

        axs[2, 1].axis('off')  
        axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

        plt.tight_layout()

        # Add main title
        plt.suptitle('pos_img_text', fontsize=16, y=1.05)
        plt.savefig('pos_img_text')
        if logger is not None:
            logger.finish()
    return results

import datetime

def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/home/tom/fsas/eeg_data/preprocessed_eeg_data", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--evaluate', action='store_true')
    # parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects        
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # visual encoder config
    embed_dim = 1024
    visual_config = {
        "image_size": 224,
        "layers": 32,
        "width": 1280,
        "head_width": 80,
        "patch_size": 14
    }

    # 1. 分布式初始化
    ddp_utils.init_distributed_mode(args)

    # 2. 设置种子 
    seed = args.seed + ddp_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 3. scaler
    scaler = torch.GradScaler()

    # 4. 主循环
    for sub in subjects:
        print("data loading")
        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        print("data loaded")
        num_tasks = ddp_utils.get_world_size()
        global_rank = ddp_utils.get_rank()
        train_sampler = ddp_utils.create_sampler([train_dataset], [True], num_tasks, global_rank)[0]
        test_sampler = None
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, sampler=test_sampler)

        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        visual_encoder = _build_vision_tower(embed_dim, visual_config)
        visual_encoder.to(device)

        if args.evaluate:
            eeg_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ATMS/sub-08/12-11_11-30/25.pth"
            visual_encoder_ckpt_path = "/home/tom/fsas/eeg_data/models/contrast/ImageEncoder/sub-08/12-11_11-30/25.pth"
            eeg_model.load_state_dict(torch.load(eeg_encoder_ckpt_path, weights_only=True))
            visual_encoder.load_state_dict(torch.load(visual_encoder_ckpt_path, weights_only=True)) 
        else:
            visual_state_dict = vlmodel.visual.state_dict()
            visual_encoder.load_state_dict(visual_state_dict) 

        # ddp
        eeg_model = torch.nn.parallel.DistributedDataParallel(eeg_model, device_ids=[device])
        visual_encoder = torch.nn.parallel.DistributedDataParallel(visual_encoder, device_ids=[device])

        optimizer = AdamW(itertools.chain(eeg_model.parameters(), visual_encoder.parameters()), lr=args.lr)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features
        preprocessed_image_cache_train_all = train_dataset.preprocessed_image_cache
        preprocessed_image_cache_test_all = test_dataset.preprocessed_image_cache

        results = main_train_loop(sub, current_time, eeg_model, visual_encoder, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all,
                                  preprocessed_image_cache_train_all, preprocessed_image_cache_test_all, config=args, logger=args.logger, scaler=scaler)


        if ddp_utils.is_main_process():
            # Save results to a CSV file
            results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
            os.makedirs(results_dir, exist_ok=True)

            if args.insubject:
                results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
            else:
                results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

            with open(results_file, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                print(f'Results saved to {results_file}')

                
if __name__ == '__main__':
    main()
