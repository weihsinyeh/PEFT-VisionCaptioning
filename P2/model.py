import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
EOS = 50256
class VITModel(nn.Module):
    def __init__(self, pretrained_model, decoder, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Encoder only use ViT-Large
        self.pretrained_model = pretrained_model

        # The embedding dimension is: 384 for ViT-S. 768 for ViT-B. 1024 for ViT-L.
        # self.Liner = nn.Linear(1280, 768).to(device)
        
        # Decoder
        self.decoder   = decoder

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -100, reduction='mean')

        self.device = device

    def forward(self, imgs, input_ids, attention_masks, gts):
        # Encoder
        feature = self.pretrained_model.forward_features(imgs)
        # feature = self.Liner(feature)
        feature = self.decoder(feature, input_ids)

        gts = torch.concat((input_ids[:, 1:], input_ids[:, :1]), dim=1)
        attention_masks = torch.concat(
            (
                attention_masks[:, 1:],
                torch.zeros(
                    (attention_masks.shape[0], 1),
                    dtype=attention_masks.dtype,
                    device=attention_masks.device,
                ),
            ),
            dim=1,
        )

        for i, mask in enumerate(attention_masks):
            for j, mask_val in enumerate(mask):
                if mask_val == 0:
                    gts[i][j] = -100
        
        if gts.size(1) < feature.size(1):
            pad_length = feature.size(1) - gts.size(1)
            padding = torch.full((gts.size(0), pad_length), -100, device=gts.device, dtype=gts.dtype)
            gts = torch.cat([gts, padding], dim=1)
        ######## Test the EOS probability
        eos_prob = torch.softmax(feature, dim=-1)[:, :, EOS].mean()
        print("EOS probability:", eos_prob)
        ########################
        feature = feature.permute(0,2,1)
        loss = self.loss_fn(feature, gts)
        return loss
    
    def generator_decoder(self, x: Tensor, encoder_feature: Tensor):
        x   = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x   = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
        x   = torch.cat([encoder_feature, x], dim=1)
        x   = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
        return x

    def generate(self, imgs, max_length=40):
        self.eval()
        # ensures that img always has a batch dimension
        if imgs.dim() < 4:
            imgs = imgs.unsqueeze(0)
        # Encoder
        with torch.no_grad():
            feature = self.pretrained_model.forward_features(imgs)
            # feature = self.Liner(feature)

        current_token = torch.tensor([[EOS]]).to(self.device)
        for i in range(max_length):
            with torch.no_grad():
                next_probability = self.generator_decoder(current_token, feature)

            next_token = next_probability.argmax(dim=-1).unsqueeze(0)
            print(next_token)
            # if next_token.item() == EOS:
            #    break

            current_token = torch.concat((current_token, next_token), dim=-1)
        return current_token[0, 1:].cpu().tolist()