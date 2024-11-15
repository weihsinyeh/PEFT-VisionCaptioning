import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from dataloader import DataLoaderTrain, DataLoaderTest

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p1_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred")
    parser.add_argument("--output_checkpoint",  type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p1_data/images/val")
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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    ############ Load Dataset ############
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation, tokenizer, augmentation)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn)

    ############ Load Model ############
    model = AutoModelForCausalLM.from_pretrained(model_name)

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"].to(device)
            batch["input_ids"]  = batch["input_ids"].to(device)
            batch["GT_ids"]     = batch["GT_ids"].to(device)

            instruction = "Generate a descriptive caption for the image."
            # Define generation parameters
            generation_config = GenerationConfig(   max_length = 50,  # Adjust based on desired caption length
                                                    num_beams = 5,  # Beam search for better results
                                                    temperature = 0.7,  # Adjust creativity vs. accuracy
                                                    top_p = 0.9,) # Nucleus sampling)

            # Convert image to appropriate format (depends on how LLaVA is implemented)
            # Assuming an image processing function exists, like `preprocess_image` for the model
            image_features = preprocess_image(image)

            # Encode instruction and prepare input
            input_ids = tokenizer.encode(instruction, return_tensors="pt")

            # Combine the instruction and image features (method may vary)
            inputs = {
                "input_ids": input_ids,
                "image_features": image_features,  # Make sure this is correctly handled by your model
            }

            # Generate the caption
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            # Decode the generated caption
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("Generated Caption:", caption)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()