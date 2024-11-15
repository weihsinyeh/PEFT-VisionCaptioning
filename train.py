import timm, argparse, torch, json
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from P2.dataloader import DataLoaderTrain, DataLoaderTest
from P2.transform import augmentation
from P2.model import VITModel
from decoder import Decoder, Config

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_annotation",    type = str,     default = "./hw3_data/p2_data/train.json")
    parser.add_argument("--valid_annotation",    type = str,     default = "./hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "./P2_output/predictions.json")
    parser.add_argument("--train_images_dir",   type = str,     default = "./hw3_data/p2_data/images/train")
    parser.add_argument("--valid_images_dir",   type = str,     default = "./hw3_data/p2_data/images/val")
    parser.add_argument("--decoder",            type = str,     default = "./hw3_data/p2_data/decoder_model.bin")
    parser.add_argument("--batch_size",         type = int,     default = 8)
    parser.add_argument("--lr",                 type = float,   default = 1e-4)
    parser.add_argument("--epochs",             type = int,     default = 100)
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ############ Load Tokenizer ############
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    ############ Load Dataset ############
    TrainDataset = DataLoaderTrain(config.train_images_dir, config.train_annotation, tokenizer, augmentation)
    train_loader = DataLoader(TrainDataset, batch_size = config.batch_size, collate_fn = TrainDataset.collate_fn)
    ValidDataset = DataLoaderTest(config.valid_images_dir)
    valid_loader = DataLoader(ValidDataset, batch_size = config.batch_size, collate_fn = ValidDataset.collate_fn)
    
    ############ Load Encoder : ViT-Large from timm ############
    pretrained_model = timm.create_model("vit_huge_patch14_clip_224.laion2b", pretrained=True).to(device)
    
    ############ Load Decoder ############
    deconder_config = Config(config.decoder)
    decoder = Decoder(deconder_config).to(device)
    
    ############ Load Model ############
    model = VITModel(pretrained_model, decoder, tokenizer).to(device)

    ############ Optimizer ############
    # Vision Transformers (ViTs) commonly use the AdamW optimizer for training. 
    # Learning rate is ften around 1e-4 to 1e-3.
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader) - 1000)

    # Freeze the decoder first
    for param in model.parameters():
        param.requires_grad = False

    for param in model.Liner.parameters():
        param.requires_grad = True
    
    # Train the cross attention
    for i in range(len(model.decoder.transformer.h)):
        for param in model.decoder.transformer.h[i].parameters():
            param.requires_grad = True

    # Recoder the weights that is trained
    train_parameter_name = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            train_parameter_name.append(name)

    for epoch in range(config.epochs):
        ################ Train ################
        train_loss = 0
        for batch in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            batch["images"]     = batch["images"].to(device)
            batch["imput_ids"]  = batch["input_ids"].to(device)

            loss = model(   batch["image"],
                            batch["input_ids"],
                            batch["GT_ids"],
                            batch["attention_masks"])

            loss.backward()
            optimizer.step()
            # Update scheduler
            scheduler.step()
            # Track loss for the epoch
            train_loss += loss.item()

            tqdm.write(f"Loss: {loss.item():.4f}")  # Optionally print the loss for each step
            tqdm.set_postfix(loss=loss.item())  # Display loss in the progress bar
        
        print(f"Epoch {epoch} Train Loss: {train_loss / len(train_loader)}")
        # Save model
        save_paremeter_weights = {}
        for key, value in model.state_dict().items():
            if key in train_parameter_name:
                save_paremeter_weights[key] = value
        
        torch.save(save_paremeter_weights, f"./P2_output/epoch_{epoch}.bin")
        ################ Evaluation ################
        model.eval()
        val_loss = 0
        output_data = {}
        for val_data in tqdm(valid_loader):
            val_data["image"]      = val_data["image"].to(device)
            output_ids = model.generate(val_data["image"])
            sentence = tokenizer.decode(output_ids)

            loss = model(   val_data["image"],
                            val_data["input_ids"],
                            val_data["GT_ids"],
                            val_data["attention_masks"])

            val_loss += loss.item()
            tqdm.write(f"Loss: {loss.item():.4f}")
            tqdm.set_postfix(loss=loss.item())
            output_data[val_data["img_name"]] = sentence

        print(f"Epoch {epoch} Validation Loss: {val_loss / len(valid_loader)}")
        # Save predictions to json
        with open(config.pred_file, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()