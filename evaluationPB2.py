import timm, argparse, torch, json, os
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from P2.dataloader import DataLoaderTrain, DataLoaderTest
from P2.transform import augmentation
from P2.model import VITModel
from decoder import Decoder, Config
import loralib as lora
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred_2")
    parser.add_argument("--output_checkpoint",  type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint_2")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--epoch",             type = int,      default = 0)
    parser.add_argument("--all_epoch",         type = bool,      default = False)
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    return parser.parse_args()

def main():
    config = parse()

    # Create directories
    if config.pred_file is not None:
        os.makedirs(config.pred_file, exist_ok=True)
    if config.output_checkpoint is not None:
        os.makedirs(config.output_checkpoint, exist_ok=True)

    # Set Device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    # Load Dataset
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation, tokenizer, augmentation)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 4, shuffle = False)
    
    # Load Encoder
    pretrained_model = timm.create_model('vit_large_patch14_clip_224', pretrained=True, num_classes=0).to(config.device)
    
    # Load Decoder
    deconder_config = Config(config.decoder)
    decoder = Decoder(deconder_config, config.device).to(config.device)
    decoder.load_state_dict(torch.load(config.decoder), strict=False)
    # Load Model
    model = VITModel(pretrained_model, decoder, tokenizer, config.device)
    
    if config.all_epoch:
        evaluation_epcoh = range(config.epoch)
    else :
        evaluation_epcoh = [config.epoch]

    for epoch in evaluation_epcoh:
              
        checkpoint_path = os.path.join(config.output_checkpoint,f"epoch_{epoch}.bin")
        model.eval()
        # Load
        checkpoint = torch.load(checkpoint_path)
        lora_params = checkpoint["lora_state_dict"]
        model.load_state_dict(lora_params, strict=False)
        model.Linear.load_state_dict(checkpoint["trainable_params"])

        output_data = {}
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"].to(config.device)
            batch["input_ids"]  = batch["input_ids"].to(config.device)
            batch["attention_masks"] = batch["attention_masks"].to(config.device)
            output_ids          = model.generate(batch["images"])
            sentence            = tokenizer.decode(output_ids)
            for i in range(len(batch["filenames"])):
                output_data[batch["filenames"][i]] = sentence

        # Save predictions to json
        file_name = f"Epoch_{epoch}.json"
        path = os.path.join(config.pred_file,file_name)
        with open(path, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()