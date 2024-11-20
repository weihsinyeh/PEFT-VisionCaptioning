import json, os, torch, timm, math, collections, argparse
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
from torchvision.utils import make_grid
import cv2
from tokenizer import BPETokenizer
from P2.dataloader import DataLoaderTrain, DataLoaderTest
from P2.transform import augmentation, transform
from P2.model import VITModel
from P2.setting import modelname
from decoder import Decoder, Config
import matplotlib.gridspec as gridspec
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred_new9")
    parser.add_argument("--checkpoint_path",    type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint_new9")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--outimg",             type = str,     default = "./outputimg")
    parser.add_argument("--epoch",              type = int,     default = 3)
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    parser.add_argument("--batch_size",         type = int,     default = 32)
    parser.add_argument("--lr",                 type = float,   default = 1e-3)
    parser.add_argument("--epochs",             type = int,     default = 100)
    parser.add_argument("--projection_dropout", type = float,   default = 0.1)
    parser.add_argument("--lora_dropout",       type = float,   default = 0.1)
    parser.add_argument("--weight_decay",       type = float,   default = 1e-3)
    parser.add_argument("--T_max",              type = int,     default = 4)
    return parser.parse_args()

def make_grid_plt(images, captions, save_path):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 5, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1, 1])

    for i, (image_path, caption) in enumerate(zip(images, captions)):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(gs[i])
        ax.imshow(img)
        ax.set_title(caption, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    plt.close()
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
    ValidDataset        = DataLoaderTest(config.valid_images_dir, transform)
    valid_loader        = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 4, shuffle = False)

    # Load Encoder
    pretrained_model    = timm.create_model(modelname, pretrained=True, num_classes=0).to(config.device)

    # Load Decoder
    decoder_config = Config(config.decoder)
    decoder_config.attention_visualization = True
    decoder = Decoder(decoder_config, config.device).to(config.device)
    decoder.load_state_dict(torch.load(config.decoder), strict=False)

    # Load Model
    model = VITModel(pretrained_model, decoder, tokenizer, device, projection_dropout = config.projection_dropout, attention_visualization = True)
    model.to(config.device)

    # Load Weight
    file_name           = os.path.join(config.checkpoint_path,f"epoch_{config.epoch}.bin")
    checkpoint          = torch.load(file_name)
    lora_params         = checkpoint["lora_state_dict"]
    trainable_params    = checkpoint["trainable_params"]
    model.load_state_dict(lora_params, strict=False)
    model.Linear.load_state_dict(trainable_params, strict=False)

    loaded_lora_params_count        = sum(p.numel() for p in lora_params.values())
    loaded_trainable_params_count   = sum(p.numel() for p in trainable_params.values())
    print(f"Loaded LoRA parameters: {loaded_lora_params_count}")
    print(f"Loaded trainable parameters: {loaded_trainable_params_count}")
    print(f"Total loaded parameters: {loaded_lora_params_count + loaded_trainable_params_count}")

    # Load Generated Captions
    path = os.path.join(config.pred_file, f"Epoch_{config.epoch}.json")


    model.eval()
    for batch in tqdm(valid_loader):
        batch["images"]             = batch["images"].to(device)
        input_ids = []

        # Get all token's attention list
        output_ids, attention_lists  = model.generate(batch["images"])
        captions = []
        captions.append("<|endoftext|>")
        for id in output_ids:
            captions.append(tokenizer.decode([id]))
        
        # Read Image
        images = []
        save_img_name = f"{0}.jpg"
        save_dir = os.path.join(config.outimg, batch["filenames"][0])
        os.makedirs(save_dir, exist_ok=True)
        save_img_path = os.path.join(save_dir, save_img_name)
        path = os.path.join(config.valid_images_dir, batch["filepaths"][0])
        origin_img = cv2.imread(path)
        cv2.imwrite(save_img_path, origin_img)
        images.append(save_img_path)
        for i, attention_list in enumerate(attention_lists): # attention map of each token
            # attention map of the last layer
            # attention_list contains 12 attention maps of each layer (12 layers)
            # Last layer is the 12th layer
            last_attention_layer = attention_list[-2] # ([1, 12, 258, 258])
            #print("last_attention_layer.shpae",last_attention_layer.shape)
            att_map     = last_attention_layer.squeeze(0) # ([12, 258, 258])
            #print("att_map.shpae",att_map.shape)
            # sum over heads
            attention_heads = []
            for attention_head in range(0, 12):
                single_att_map     = att_map[attention_head] # ([258, 258])
                print("att_map.shpae",single_att_map.shape)
                last_token_att_map = single_att_map[256+i, 1:257] # shape: [1, 12, 258]

                # normalize attention map
                original = last_token_att_map
                last_token_att_map = last_token_att_map * 1e3
                min_val = last_token_att_map.min()
                max_val = last_token_att_map.max()
                last_token_att_map = ( (last_token_att_map - min_val) * 255 / (max_val - min_val) )           

                att_map_reshaped = [[0 for _ in range(16)] for _ in range(16)]
                att_map_reshaped = last_token_att_map.view(16, 16).cpu().detach().numpy().astype(np.uint8)
                
                # use att_map_reshaped to draw 16x16 attention map
                grid_width = origin_img.shape[1] // 16
                grid_height = origin_img.shape[0] // 16

                # Initialize the attention map for the larger image
                att_map_expanded = np.zeros((origin_img.shape[0], origin_img.shape[1]), dtype=np.uint8)

                # Fill the expanded attention map with values from att_map_reshaped
                for k in range(16):
                    for j in range(16):
                        # Calculate the region in the larger image corresponding to each value in att_map_reshaped
                        x_start = k * grid_width
                        x_end = (k + 1) * grid_width
                        y_start = j * grid_height
                        y_end = (j + 1) * grid_height
                    
                        # Fill this region with the corresponding value from att_map_reshaped
                        att_map_expanded[y_start:y_end, x_start:x_end] = att_map_reshaped[k, j]
                attention_heads.append(att_map_expanded)
            # Sum the attention maps from all heads
            att_map_expanded = np.sum(attention_heads, axis=0)
            att_map_expanded = (att_map_expanded - att_map_expanded.min()) / (att_map_expanded.max() - att_map_expanded.min())
            att_map_expanded = (att_map_expanded * 255).astype(np.uint8)
            # Optionally, you can overlay this on the original image
            # Here, I'm just using it as a background for demonstration
            att_map_expanded = cv2.resize(att_map_expanded, (origin_img.shape[1], origin_img.shape[0]))
            save_dir = os.path.join(config.outimg, batch["filenames"][0])
            os.makedirs(save_dir, exist_ok=True)
            save_img_name = f"{i+1}.jpg"
            save_img_path = os.path.join(save_dir, save_img_name)
            cv2.imwrite(save_img_path, att_map_expanded)
            # width = sqrt_len
            # att_mask = att_map_reshaped.reshape(width, width)

           
            heatmap = cv2.applyColorMap(att_map_expanded , cv2.COLORMAP_JET)

            result = cv2.addWeighted(heatmap, 0.5, origin_img, 0.5, 0)

            save_dir = os.path.join(config.outimg, batch["filenames"][0])
            os.makedirs(save_dir, exist_ok=True)
            save_img_name = f"{batch['filenames'][0]}_{i+1}.jpg"
            save_img_path = os.path.join(save_dir, save_img_name)
            cv2.imwrite(save_img_path, result)
            images.append(save_img_path)
        captions.append("<|endoftext|>")
        make_grid_plt(images, captions, f'{save_dir}/{batch["filenames"][0]}.jpg')

if __name__ == "__main__":
    main()