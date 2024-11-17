import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F
BOS = 50256
EOS = 50256
class VITModel(nn.Module):
    def __init__(self, pretrained_model, decoder, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Encoder only use ViT-Large
        self.pretrained_model = pretrained_model

        # The embedding dimension is: 384 for ViT-S. 768 for ViT-B. 1024 for ViT-L.
        self.Linear = nn.Linear(1024, 768).to(device) 
        
        # Decoder
        self.decoder   = decoder

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -100)

        self.device = device

    def forward(self, imgs, input_ids, attention_masks):
        # Encoder
        feature = self.pretrained_model.forward_features(imgs)
        # print("Raw feature from encoder:", feature)
        feature = self.Linear(feature)

        feature = self.decoder(feature, input_ids)

        gts = torch.concat((input_ids[:, 1:], input_ids[:, :1]), dim=1)
        # print("feature:",feature)
        #print("input_ids",input_ids[0])
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

        gts[attention_masks == 0] = -100
        
        if gts.size(1) < feature.size(1):
            pad_length = feature.size(1) - gts.size(1)
            padding = torch.full((gts.size(0), pad_length), -100, device=gts.device, dtype=gts.dtype)
            gts = torch.cat([gts, padding], dim=1)
 
        feature = feature.permute(0,2,1)
        loss = self.loss_fn(feature, gts)
        return loss
    
    def generator_decoder(self, x: Tensor, visual_embeds: Tensor):
        x   = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x   = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
        x   = torch.cat([visual_embeds, x], dim=1)
        x   = self.decoder.transformer.h(x)
        x   = self.decoder.transformer.ln_f(x)
        text_output = x[:, -1]
        x   = self.decoder.lm_head(text_output)
        return x

    def generate(self, imgs):
        self.eval()
        # ensures that img always has a batch dimension
        if imgs.dim() < 4:
            imgs = imgs.unsqueeze(0)
        # Encoder
        feature = self.pretrained_model.forward_features(imgs)
        feature = self.Linear(feature)
        output  = self.beamsearch(feature)
        return output
    
    def greedy_search(self, img, max_length=30):
        
        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            encoder_feature = self.encoder.forward_features(img)
            encoder_feature = self.feature_resize(encoder_feature)

        cur_state = torch.tensor([EOS]).to(device).unsqueeze(1)
        for _ in range(max_length):
            with torch.no_grad():
                next_prob = generator_decoder(cur_state, encoder_feature)

            next_word = next_prob.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == EOS:
                break
            cur_state = torch.concat((cur_state, next_word), dim=-1)
        return cur_state[0, 1:].cpu().tolist()  # remove [BOS]

    def beamsearch(self, feature, beams=4, max_length=30):
        cur_token = torch.tensor([BOS]).to(self.device).unsqueeze(1)
        next_p = self.generator_decoder(cur_token, feature)
        vocab_size = next_p.shape[-1]

        # Debug: Check initial probabilities
        # print(f"Initial next_p shape: {next_p.shape}, vocab_size: {vocab_size}")

        cur_p, next_token = next_p.log_softmax(-1).topk(k=beams, axis=-1)
        cur_p = cur_p.reshape(beams)
        next_token = next_token.reshape(beams, 1)

        cur_token = cur_token.repeat((beams, 1))
        cur_token = torch.cat((cur_token, next_token), axis=1)

        ans_ids = []
        ans_probs = []

        for i in range(max_length - 1):
            # Get top k beams for beam*beam candidates
            next_p = self.generator_decoder(
                cur_token, feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_p = cur_p.unsqueeze(-1) + next_p
            cur_p = cur_p.flatten()

            # Debug: Check probabilities before normalization
            # print(f"Step {i}, cur_p shape: {cur_p.shape}")

            # Length normalization
            _, idx = (cur_p / (len(cur_token[0]) + 1)).topk(k=beams, dim=-1)
            cur_p = cur_p[idx]

            # Get corresponding next char
            next_token = torch.remainder(idx, vocab_size)
            next_token = next_token.unsqueeze(-1)

            # Get corresponding original beams
            top_candidates = (idx / vocab_size).long()
            cur_token = cur_token[top_candidates]
            cur_token = torch.cat((cur_token, next_token), dim=1)

            # Debug: Output current token states
            # print(f"Step {i}, cur_token: {cur_token.shape}")

            # Check if we should finalize a beam
            to_rm_idx = set()
            for idx, ch in enumerate(next_token):
                if i == (max_length - 2) or ch.item() == EOS:
                    ans_ids.append(cur_token[idx].cpu().tolist())
                    ans_probs.append(cur_p[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_token)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break

            cur_token = cur_token[to_keep_idx]
            cur_p = cur_p[to_keep_idx]

        # Get the best answer based on probability
        if ans_probs:
            max_idx = torch.argmax(torch.tensor(ans_probs)).item()
            ans_ids[max_idx] = [x for x in ans_ids[max_idx] if x != EOS]
            # print("ans_ids[max_idx]", ans_ids[max_idx])
            return ans_ids[max_idx]
        else:
            print("No valid sequence generated.")
            return []