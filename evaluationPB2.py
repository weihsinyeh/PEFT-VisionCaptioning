import timm, argparse, torch, json, os
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from P2.dataloader import DataLoaderTrain, DataLoaderTest
from P2.transform import augmentation, transform
from P2.model import VITModel
from P2.setting import modelname
from decoder import Decoder, Config
import loralib as lora
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred_5")
    parser.add_argument("--checkpoint",         type = str,     default = "./epoch_3.bin")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    return parser.parse_args()

def main():
    config = parse()

    # Save Prediction
    print(f"Prediction files will be saved to {config.pred_file}")

    # Set Device
    config.device       = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer
    tokenizer           = BPETokenizer("encoder.json", "vocab.bpe")

    # Load Dataset
    ValidDataset        = DataLoaderTest(config.valid_images_dir, transform)
    valid_loader        = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 1, shuffle = False)
    
    # Load Encoder
    pretrained_model    = timm.create_model(modelname, pretrained=True, num_classes = 0).to(config.device)
    print(f"Pretrained model: {modelname}")

    # Load Decoder
    deconder_config     = Config(config.decoder)
    decoder             = Decoder(deconder_config, config.device).to(config.device)
    decoder.load_state_dict(torch.load(config.decoder), strict=False)

    # Load Model
    model = VITModel(pretrained_model, decoder, tokenizer, config.device)
    model.to(config.device)
              
    checkpoint_path = os.path.join(config.checkpoint)
    model.eval()
    # Load
    checkpoint          = torch.load(config.checkpoint)
    lora_params         = checkpoint["lora_state_dict"]
    trainable_params    = checkpoint["trainable_params"]
    model.load_state_dict(lora_params, strict=False)
    model.Linear.load_state_dict(trainable_params)

    loaded_lora_params_count = sum(p.numel() for p in lora_params.values())
    loaded_trainable_params_count = sum(p.numel() for p in trainable_params.values())
    print(f"Loaded LoRA parameters: {loaded_lora_params_count}")
    print(f"Loaded trainable parameters: {loaded_trainable_params_count}")

    output_data = {}
    for batch in tqdm(valid_loader):
        batch["images"]     = batch["images"].to(config.device)
        output_ids          = model.generate(batch["images"])
        sentence            = tokenizer.decode(output_ids)
        for i in range(len(batch["filenames"])):
            output_data[batch["filenames"][i]] = sentence

    # Save predictions to json
    with open(config.pred_file, "w") as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()