import torch
import torch.nn as nn
import torch.nn.functional as F
class VITModel(nn.Module):
    def __init__(self, pretrained_model, decoder, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Encoder only use ViT-Large
        self.pretrained_model = pretrained_model

        # The embedding dimension is: 384 for ViT-S. 768 for ViT-B. 1024 for ViT-L.
        self.Liner = nn.Linear(1024, 768)
        
        # Decoder
        self.decoder   = decoder

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -100)


    def forward(self, imgs, input_ids, gts, attention_masks):
        # Encoder
        feature = self.encoder.forward_features(imgs)
        attention_masks = torch.concat((attention_masks[:, 1:],
                                        torch.zeros((   attention_masks.shape[0], 1),
                                                        dtype=attention_masks.dtype,
                                                        device=attention_masks.device)),dim=1,)
        feature = self.Liner(feature)
        feature = self.decoder(feature, input_ids)
        print("feature", feature.shape)
        print("gts", gts.shape)
        self.loss = self.loss_fn(feature.view(-1, feature.size(-1)), gts.view(-1))
        return self.loss