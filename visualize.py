import json, os, torch, timm, math, collections
from torch import Tensor, nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from tokenizer import BPETokenizer
from P2.dataloader import DataLoaderTrain, DataLoaderTest
from P2.transform import augmentation, transform
from P2.model import VITModel
from P2.setting import modelname
from decoder import Decoder, Config

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred_new")
    parser.add_argument("--checkpoint_path",    type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint_new/epoch_0.bin")

    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--outimg",             type = str,     default = "./outputimg")
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    return parser.parse_args()


def visualize_attention(img, querys, keys, output_ids, img_name):
    tokenizer = BPETokenizer()
    img = img.squeeze(0).permute(1, 2, 0).cpu()
    img = (img - img.min()) / (img.max() - img.min())

    num_cols = 4  # 每row兩個
    num_plots = len(querys)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(len(querys)):
        ax = axes[i // num_cols, i % num_cols]
        title = tokenizer.decode([output_ids[i]]) # q = [b,tgt l,768] k = [b,src l ,768] -> q@k.trans -> [tgt l , src l] -> [1,16,16] -> 224,224
        att = querys[i] @ keys[i].permute(1, 0) * (1.0 / math.sqrt(keys[i].size(-1)))
        att = att[-1, 1:].view(1, 16, 16)
        attention_resized = F.interpolate(
            att.unsqueeze(0), size=img.shape[:2], mode="bilinear", align_corners=False
        )

        # plt.imshow(attention_resized, cmap="jet", alpha=0.5)  # 使用半透明的熱圖

        ax.imshow(img.cpu())
        ax.set_title(f"{title}")

        if i != 0:
            ax.imshow(attention_resized.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)

    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    # plt.colorbar()
    plt.savefig(f"{img_name}.png")

attention_weights = []
def fetch_attention_weights(module, input, output):
    attention_weights.append(output[1].detach().cpu())

def main():
    config = parse()
    # Create directories
    if config.outimg is not None:
        os.makedirs(config.outimg, exist_ok=True)
        print(f"Output img will be saved to {config.outimg}")

    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    # Load Tokenizer
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    # Load Dataset
    ValidDataset        = DataLoaderTrain(config.valid_images_dir, config.valid_annotation, tokenizer, transform)
    valid_loader        = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 4, shuffle = False)

    # Load Encoder
    pretrained_model    = timm.create_model(modelname, pretrained=True, num_classes=0).to(config.device)

    # Load Decoder
    deconder_config = Config(config.decoder)
    decoder = Decoder(deconder_config, config.device).to(config.device)
    decoder.load_state_dict(torch.load(config.decoder), strict=False)

    # Load Model
    model = VITModel(pretrained_model, decoder, tokenizer, config.device)
    model.to(config.device)

    # Load Weight
    checkpoint          = torch.load(config.checkpoint_path)
    lora_params         = checkpoint["lora_state_dict"]
    trainable_params    = checkpoint["trainable_params"]
    model.load_state_dict(lora_params, strict=False)
    model.Linear.load_state_dict(trainable_params)

    loaded_lora_params_count        = sum(p.numel() for p in lora_params.values())
    loaded_trainable_params_count   = sum(p.numel() for p in trainable_params.values())
    print(f"Loaded LoRA parameters: {loaded_lora_params_count}")
    print(f"Loaded trainable parameters: {loaded_trainable_params_count}")
    print(f"Total loaded parameters: {loaded_lora_params_count + loaded_trainable_params_count}")

    model.eval()
    for val_data in tqdm(val_loader):
        hook_q = []
        hook_k = []
        for block in model.decoder.transformer.h:
            block.attn.register_forward_hook(fetch_attention_weights)

        ori_img, img, filename = val_data
        img = img.to(device)

        with torch.autocast(device_type="cuda"):
            output_ids = model.generate(batch["images"])

        output_ids.insert(0, EOS_TOKEN)
        output_ids.insert(len(output_ids), EOS_TOKEN)
        tokenizer = BPETokenizer()
        print(tokenizer.decode(output_ids))

        querys  = []
        keys    = []
        querys  = [hook_q[i] for i in range(1, len(hook_q)) if i % 11 == 0]
        keys    = [hook_k[i] for i in range(1, len(hook_k)) if i % 11 == 0]

        visualize_attention(    ori_img,
                                querys[: len(output_ids)],
                                keys[: len(output_ids)],
                                output_ids,
                                filename,)

if __name__ == "__main__":
    main()
