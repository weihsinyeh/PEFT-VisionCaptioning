import torch, argparse, transformers, tqdm, os, json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p1_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P1_pred")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p1_data/images/val")
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Check pred directory exist
    if config.pred_file is not None:
        os.makedirs(config.pred_file, exist_ok=True)
        print(f"Prediction files will be saved to {config.pred_file}")

    ############ Load Tokenizer ############
    processor = AutoProcessor.from_pretrained(model_id)

    ############ Load Dataset ############
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn)
    ############ Load Model ############
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    outputs = {}
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"]
            batch["captions"]   = batch["captions"]
 
            inputs = processor(images=batch["images"], text=prompt, return_tensors='pt').to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False, output_attention = True)
            generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
            for text in generated_text:
                outputs[batch["filenames"][0]] = text.split("ASSISTANT:")[-1]
            break
    files = os.listdir(config.pred_file)
    
    # 過濾出以 'pred_' 開頭並且以 '.json' 結尾的檔案
    pred_files = [file for file in files if file.startswith("pred_") and file.endswith(".json")]
    path = os.path.join(config.pred_file, f"pred_{len(pred_files)}.json")
    with open(config.pred_file, "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()