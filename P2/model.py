import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
BOS = 50256
EOS = 50256
class VITModel(nn.Module):
    def __init__(self, pretrained_model, decoder, tokenizer, device, projection_dropout, attention_visualization = False):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Encoder only use ViT-Large
        self.pretrained_model = pretrained_model

        # The embedding dimension is: 384 for ViT-S. 768 for ViT-B. 1024 for ViT-L.
        self.Linear = nn.Sequential(
            nn.Dropout(projection_dropout),
            nn.Linear(1664, 768).to(device),
            nn.Dropout(projection_dropout)
        )
        
        # Decoder
        self.decoder   = decoder.to(device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean',ignore_index = -100)

        # Device
        self.device = device

        # Visualize Attention
        self.attention_visualization = attention_visualization

    def forward(self, imgs, input_ids, attention_masks):
        # Encoder
        feature = self.pretrained_model.forward_features(imgs)
        
        feature = self.Linear(feature)

        feature = self.decoder(feature, input_ids)

        gts = torch.concat((input_ids[:, 1:], input_ids[:, :1]), dim=1)
        gts[attention_masks == 0] = -100
        if gts.size(1) < feature.size(1):
            pad_length = feature.size(1) - gts.size(1)
            padding = torch.full((gts.size(0), pad_length), -100, device=gts.device, dtype=gts.dtype)
            gts = torch.cat([gts, padding], dim=1)
        elif gts.size(1) > feature.size(1):
            gts = gts[:, :feature.size(1)]
        feature = feature.permute(0,2,1)
        loss = self.loss_fn(feature, gts)
        return loss

    def generate(self, imgs):
        # Encoder
        feature = self.pretrained_model.forward_features(imgs)
        feature = self.Linear(feature)
        if self.attention_visualization == False:
            outputs = self.greedy_search(feature)
        else:
            outputs, attention_list = self.greedy_search(feature)
        # Remove BOS
        if outputs[0] == BOS:
            outputs = outputs[1:]
        for i in range(len(outputs)):
            if outputs[i] == EOS:
                outputs = outputs[:i]
                break
        if self.attention_visualization == False:
            return outputs
        else:
            return outputs, attention_list
    
    def greedy_search(self, feature, max_length = 30):
        cur_token = torch.tensor([BOS], dtype=torch.long).to(self.device).unsqueeze(1)
        if self.attention_visualization:
            attention_list = []
        for i in range(max_length):
            if self.attention_visualization == False:
                next_prob = self.decoder(feature, cur_token)
            else:
                next_prob, att = self.decoder(feature, cur_token)
                attention_list.append(att)
            next_prob   = next_prob[:, -1,:]
            next_token  = next_prob.argmax(dim=-1).unsqueeze(1)
            if next_token.item() == EOS:
                break
            cur_token = torch.concat((cur_token, next_token), dim=-1)
            cur_token = cur_token.to(self.device)
        if self.attention_visualization == False:
            return cur_token[0].cpu().tolist()
        else:
            return cur_token[0].cpu().tolist(), attention_list