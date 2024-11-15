import timm, argparse, torch, json, os
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
    parser.add_argument("--train_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/train.json")
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred")
    parser.add_argument("--output_checkpoint",  type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint")
    parser.add_argument("--train_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/train")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    parser.add_argument("--batch_size",         type = int,     default = 8)
    parser.add_argument("--lr",                 type = float,   default = 1e-4)
    parser.add_argument("--epochs",             type = int,     default = 100)
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    ############ Load Tokenizer ############
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    ############ Load Dataset ############
    TrainDataset = DataLoaderTrain(config.train_images_dir, config.train_annotation, tokenizer, augmentation)
    train_loader = DataLoader(TrainDataset, batch_size = config.batch_size, collate_fn = TrainDataset.collate_fn)
    # ValidDataset = DataLoaderTest(config.valid_images_dir, augmentation)
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation, tokenizer, augmentation)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn)
    
    ############ Load Encoder : ViT-Large from timm ############
    pretrained_model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True).to(device)
    
    ############ Load Decoder ############
    deconder_config = Config(config.decoder)
    decoder = Decoder(deconder_config, config.device).to(device)
    
    ############ Load Model ############
    model = VITModel(pretrained_model, decoder, tokenizer, device)

    ############ Optimizer ############
    # Vision Transformers (ViTs) commonly use the AdamW optimizer for training. 
    # Learning rate is ften around 1e-4 to 1e-3.
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader) - 1000)

    # Freeze the all model first
    for param in model.parameters():
        param.requires_grad = False

    # for param in model.Liner.parameters():
    #    param.requires_grad = True
    
    # Train the cross attention
    for param in model.decoder.lm_head.parameters():
        param.requires_grad = True

    # Recoder the weights that is trained
    train_parameter_name = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            train_parameter_name.append(name)
    print("Total parms:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for epoch in range(config.epochs):
        ################ Train ################
        train_loss = 0
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch["images"]             = batch["images"].to(device)
            batch["input_ids"]          = batch["input_ids"].to(device)
            batch["GT_ids"]             = batch["GT_ids"].to(device)
            batch["attention_masks"]    = batch["attention_masks"].to(device)
            loss = model(   batch["images"],
                            batch["input_ids"],
                            batch["attention_masks"],
                            batch["GT_ids"])

            loss.backward()
            optimizer.step()
            # Update scheduler
            scheduler.step()
            # Track loss for the epoch
            train_loss += loss.item()
        print(f"Epoch {epoch} Train Loss: {train_loss / len(train_loader)}")
        # Save model
        save_paremeter_weights = {}
        for key, value in model.state_dict().items():
            if key in train_parameter_name:
                print(key)
                save_paremeter_weights[key] = value
        
        checpoint_path = os.path.join(config.output_checkpoint,f"epoch_{epoch}.bin")
        torch.save(save_paremeter_weights, checpoint_path)
        ################ Evaluation ################
        model.eval()
        val_loss = 0
        output_data = {}
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"].to(device)
            batch["input_ids"]  = batch["input_ids"].to(device)
            batch["GT_ids"]     = batch["GT_ids"].to(device)

            output_ids          = model.generate(batch["images"])
            sentence            = tokenizer.decode(output_ids)
            with torch.no_grad():
                loss = model(   batch["images"],
                                batch["input_ids"],
                                batch["GT_ids"])

            val_loss += loss.item()
            for i in range(len(batch["filenames"])):
                print(f"{batch['filenames'][i]}: {sentence}")
                output_data[batch["filenames"][i]] = sentence

        print(f"Epoch {epoch} Validation Loss: {val_loss / len(valid_loader)}")
        # Save predictions to json
        file_name = f"Epoch_{epoch}.json"
        path = os.path.join(config.pred_file,file_name)
        with open(path, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()