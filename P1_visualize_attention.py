import json, os, torch, collections, argparse
from torch import Tensor, nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid
import cv2
import matplotlib.gridspec as gridspec
import torch, argparse, transformers, os, json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from P1.dataloader import DataLoaderTest
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p3_data/images")
    parser.add_argument("--outimg",             type = str,     default = "./outputimgP1")
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
        print("caption",caption)
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


    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Load Tokenizer
    processor = AutoProcessor.from_pretrained(model_id)

    # Load Dataset
    ValidDataset        = DataLoaderTest(config.valid_images_dir)
    valid_loader        = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 4, shuffle = False)
    # Load Model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    # Best : "Focus only on the primary object and its core action in this image. Keep the caption short and clear. Avoid any extra details."
    conversation = [
        {
            "role": "user",
            "content": [                  
                {"type": "text", "text":  "Focus only on the primary subject and its core action in this image. Keep the caption short and clear. Avoid any extra details."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    model.eval()
    for batch in tqdm(valid_loader):
        # batch["images"]             = batch["images"].to(device)
        inputs = processor(images=batch["images"], text=prompt, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, output_attentions=True, return_dict_in_generate=True)
        generated_text = processor.decode(output[0][2:], skip_special_tokens=True)

        print("output.keys()",output.keys())
        print("sequence_output.shape",output["sequences"].shape)

        attentions = output["attentions"]
        captions = []
        image_start, image_end, text_start, text_end = -1, -1, -1, len(output["sequences"][0])
        for id in range(5, len(output["sequences"][0])+1):
            if output["sequences"][0][id-1:id] == 32000:
                if(image_start == -1):
                    image_start = id
                continue
            if (image_start != -1 and image_end == -1):
                image_end = id
            if output["sequences"][0][id-1:id] == 29901:
                text_start = id
                continue
            captions.append(processor.decode(output["sequences"][0][id-1:id], skip_special_tokens=True))
        captions = []
        captions.append("<|endoftext|>")
        for id in range(text_start, len(output["sequences"][0])):
            captions.append(processor.decode(output["sequences"][0][id:id+1], skip_special_tokens=True))
        
        print("captions",captions)
        print("image_start",image_start)
        print("image_end",image_end)
        print("text_start",text_start)
        print("text_end",text_end)
        # Read Image
        path = os.path.join(config.valid_images_dir, batch["filepaths"][0])
        origin_img = cv2.imread(path)
        images = []

        print("attentions",len(attentions)) # 14 text token
        save_dir = os.path.join(config.outimg, batch["filenames"][0])
        os.makedirs(save_dir, exist_ok=True)
        save_img_name = f"{batch['filenames'][0]}_0.jpg"
        save_img_path = os.path.join(save_dir, save_img_name)
        cv2.imwrite(save_img_path, origin_img)
        images.append(save_img_path)
        captions.append("<|endoftext|>")
        for i, attention_list in enumerate(attentions): # attention map of each token
            # 32 heads
            new_attention_list = attentions[i]
            # print("new_attention_list.shpae : [1, 32, text_start, text_start]",new_attention_list.shape) # ([1, 32, 632, 632])
            new_attention_list = new_attention_list[0][0] # ([32, 632, 632])
            # print("new_attention_list.shpae",new_attention_list.shape)
            att_map     = new_attention_list[:,0,image_start:image_end] 
            # print("att_map.shpae",att_map.shape) # shape: [32, 576]
            # sum over heads
            attention_heads = []
            for attention_head in range(0, 1):
                single_att_map     = att_map[attention_head][:] # ([1, 576])
                # print("single_att_map.shpae",single_att_map.shape)
                # normalize attention map
                original = single_att_map
                single_att_map = single_att_map
                min_val = single_att_map.min()
                max_val = single_att_map.max()
                single_att_map = ( (single_att_map - min_val) * 255 / (max_val - min_val) )           

                att_map_reshaped = [[0 for _ in range(24)] for _ in range(24)]
                att_map_reshaped = single_att_map.view(24, 24).cpu().detach().numpy().astype(np.uint8)
                
                # use att_map_reshaped to draw 16x16 attention map
                grid_width = origin_img.shape[1] // 24
                grid_height = origin_img.shape[0] // 24

                # Initialize the attention map for the larger image
                att_map_expanded = np.zeros((origin_img.shape[0], origin_img.shape[1]), dtype=np.uint8)

                # Fill the expanded attention map with values from att_map_reshaped
                for k in range(24):
                    for j in range(24):
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
            # print("att_map_expanded.shpae",att_map_expanded.shape)
            att_map_expanded = (att_map_expanded - att_map_expanded.min()) / (att_map_expanded.max() - att_map_expanded.min())
            att_map_expanded = (att_map_expanded * 255).astype(np.uint8)
       
            att_map_expanded = cv2.resize(att_map_expanded, (origin_img.shape[1], origin_img.shape[0]))
            save_dir = os.path.join(config.outimg, batch["filenames"][0])
            os.makedirs(save_dir, exist_ok=True)
            save_img_name = f"{i}.jpg"
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