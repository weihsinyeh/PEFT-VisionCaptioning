import torch, argparse, transformers, tqdm, os, json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from dataloader import DataLoaderTest
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file",  type = str, default = "/project/g/r13922043/hw3_output/P1_pred/pred_50.json")
    parser.add_argument("--images_dir", type = str, default = "/project/g/r13922043/hw3_data/p1_data/images/val")
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Load Tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    # Load Dataset
    TestDataset = DataLoaderTest(config.images_dir)
    test_loader = DataLoader(TestDataset, batch_size = 1, collate_fn = TestDataset.collate_fn)
    # Load Model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    # Best : "Focus only on the primary object and its core action in this image. Keep the caption short and clear. Avoid any extra details."
    conversation = [{
            "role": "user",
            "content": [                   
                {"type": "text", "text": "Focus only on the primary object and its core action in this image. Keep the caption short and clear. Avoid any extra details."},
                {"type": "image"},],},]
    print("Prompt:", conversation)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    outputs = {}
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch["images"]     = batch["images"]
            inputs = processor(images=batch["images"], text=prompt, return_tensors='pt').to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
            outputs[batch["filenames"][0]] = generated_text.split("ASSISTANT: ")[1]
    
    with open(config.pred_file, "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()