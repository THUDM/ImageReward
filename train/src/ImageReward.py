'''
@File       :   ImageReward.py
@Time       :   2023/02/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model for reward model.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from config.options import *
from config.utils import *
from models.blip_pretrain import blip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)


class ImageReward(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.blip = blip_pretrain(pretrained=config['blip_path'], image_size=config['BLIP']['image_size'], vit=config['BLIP']['vit'])
        self.preprocess = _transform(config['BLIP']['image_size'])
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        
        if opts.fix_base:
            self.blip.requires_grad_(False)
        
        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break


    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
        
        # encode data
        if opts.rank_pair:
            batch_data = self.encode_pair(batch_data)
        else:
            batch_data = self.encode_data(batch_data)
        
        # forward
        emb_better, emb_worse = batch_data['emb_better'], batch_data['emb_worse']
        
        reward_better = self.mlp(emb_better)
        reward_worse = self.mlp(emb_worse)
        reward = torch.concat((reward_better, reward_worse), dim=1)
        
        return reward


    def encode_pair(self, batch_data):
        text_ids, text_mask, img_better, img_worse = batch_data['text_ids'], batch_data['text_mask'], batch_data['img_better'], batch_data['img_worse']
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        img_better = img_better.to(self.device) # [batch_size, C, H, W]
        img_worse = img_worse.to(self.device) # [batch_size, C, H, W]
        
        # encode better emb
        image_embeds_better = self.blip.visual_encoder(img_better)
        image_atts_better = torch.ones(image_embeds_better.size()[:-1], dtype=torch.long).to(self.device)
        emb_better = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds_better,
                                            encoder_attention_mask = image_atts_better,
                                            return_dict = True,
                                           ).last_hidden_state # [batch_size, seq_len, feature_dim]
        emb_better = emb_better[:, 0, :].float()
        
        # encode worse emb
        image_embeds_worse = self.blip.visual_encoder(img_worse)
        image_atts_worse = torch.ones(image_embeds_worse.size()[:-1], dtype=torch.long).to(self.device)
        emb_worse = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds_worse,
                                            encoder_attention_mask = image_atts_worse,
                                            return_dict = True,
                                           ).last_hidden_state
        emb_worse = emb_worse[:, 0, :].float()
        
        # get batch data
        batch_data = {
            'emb_better': emb_better,
            'emb_worse': emb_worse,
        }

        return batch_data


    def encode_data(self, batch_data):
        
        txt_better, txt_worse = [], []

        for item in batch_data:
            
            text_input = self.blip.tokenizer(item["prompt"], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
            
            txt_set = []
            for generations in item["generations"]:
                # image encode
                img_path = os.path.join(config['image_base'], generations)
                pil_image = Image.open(img_path)
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_embeds = self.blip.visual_encoder(image)
                
                # text encode cross attention with image
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
                text_output = self.blip.text_encoder(text_input.input_ids,
                                                     attention_mask = text_input.attention_mask,
                                                     encoder_hidden_states = image_embeds,
                                                     encoder_attention_mask = image_atts,
                                                     return_dict = True,
                                                    )
                txt_set.append(text_output.last_hidden_state[:,0,:])
                
            labels = item["ranking"]
            for id_l in range(len(labels)):
                for id_r in range(id_l+1, len(labels)):
                    if labels[id_l] < labels[id_r]:
                        txt_better.append(txt_set[id_l])
                        txt_worse.append(txt_set[id_r])
                    elif labels[id_l] > labels[id_r]:
                        txt_better.append(txt_set[id_r])
                        txt_worse.append(txt_set[id_l])

        # torch.Size([sample_num, feature_dim])
        txt_better = torch.cat(txt_better, 0).float()
        txt_worse = torch.cat(txt_worse, 0).float()
        
        batch_data = {
            'emb_better': txt_better,
            'emb_worse': txt_worse,
        }

        return batch_data

