import torch, argparse, transformers, tqdm, os, json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_annotation",   type = str,     default = "./hw3_data/p1_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "./hw3_output/P2_pred")
    parser.add_argument("--output_checkpoint",  type = str,     default = "./hw3_output/P2_checkpoint")
    parser.add_argument("--valid_images_dir",   type = str,     default = "./hw3_data/p1_data/images/val")
    parser.add_argument("--batch_size",         type = int,     default = 8)
    parser.add_argument("--lr",                 type = float,   default = 1e-4)
    parser.add_argument("--epochs",             type = int,     default = 100)
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    model_id = "llava-hf/llava-1.5-7b-hf"

    ############ Load Tokenizer ############
    processor = AutoProcessor.from_pretrained(model_id)

    ############ Load Dataset ############
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn)
    ############ Load Model ############
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = LlavaForConditionalGeneration.from_pretrained(  model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False).to(device)

    prompts = ["USER: <image>\nPlease describe this image\nASSISTANT:",]


    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"].to(device)
            batch["captions"]   = batch["captions"]
 
            inputs = processor(prompts, batch["images"], padding=True, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
            for text in generated_text:
                print(text.split("ASSISTANT:")[-1])


if __name__ == '__main__':
    torch.manual_seed(42)
    main()

# Reference : https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Inference_with_LLaVa_for_multimodal_generation.ipynb