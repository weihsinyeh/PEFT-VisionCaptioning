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
    parser.add_argument("--train_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/train.json")
    parser.add_argument("--valid_annotation",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred_new12")
    parser.add_argument("--output_checkpoint",  type = str,     default = "/project/g/r13922043/hw3_output/P2_checkpoint_new12")
    parser.add_argument("--train_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/train")
    parser.add_argument("--valid_images_dir",   type = str,     default = "/project/g/r13922043/hw3_data/p2_data/images/val")
    parser.add_argument("--decoder",            type = str,     default = "./decoder_model.bin")
    parser.add_argument("--batch_size",         type = int,     default = 32)
    parser.add_argument("--lr",                 type = float,   default = 0.001)
    parser.add_argument("--epochs",             type = int,     default = 10)
    parser.add_argument("--projection_dropout", type = float,   default = 0.1)
    parser.add_argument("--lora_dropout",       type = float,   default = 0.1)
    parser.add_argument("--weight_decay",       type = float,   default = 0.005)
    parser.add_argument("--T_max",              type = int,     default = 5)
    return parser.parse_args()

def main():
    config = parse()
    print("Batch size:", config.batch_size)
    print("Learning rate:", config.lr)
    print("Projection dropout:", config.projection_dropout)
    print("LoRA dropout:", config.lora_dropout)
    print("Weight decay:", config.weight_decay)
    print("T_max:", config.T_max)
    # Create directories
    if config.pred_file is not None:
        os.makedirs(config.pred_file, exist_ok=True)
        print(f"Prediction files will be saved to {config.pred_file}")
    if config.output_checkpoint is not None:
        os.makedirs(config.output_checkpoint, exist_ok=True)
        print(f"Checkpoint files will be saved to {config.output_checkpoint}")

    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    # Load Tokenizer
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    # Load Dataset
    TrainDataset = DataLoaderTrain(config.train_images_dir, config.train_annotation, tokenizer, augmentation)
    train_loader = DataLoader(TrainDataset, batch_size = config.batch_size, collate_fn = TrainDataset.collate_fn, num_workers = 8, shuffle = True)
    ValidDataset = DataLoaderTrain(config.valid_images_dir, config.valid_annotation, tokenizer, transform)
    valid_loader = DataLoader(ValidDataset, batch_size = 1, collate_fn = ValidDataset.collate_fn, num_workers = 8, shuffle = False)
    
    # Load Encoder
    pretrained_model = timm.create_model(modelname, pretrained=True, num_classes=0).to(device)
    print(f"Pretrained model: {modelname}")
    
    # Load Decoder
    deconder_config = Config(config.decoder)
    deconder_config.dropout = config.lora_dropout
    decoder = Decoder(deconder_config, config.device).to(device)
    # Load Model
    model = VITModel(pretrained_model, decoder, tokenizer, device, projection_dropout = config.projection_dropout, attention_visualization = False)

    # Set Trainable parameters
    lora.mark_only_lora_as_trainable(model)

    # Unfreeze the Projection layer
    for param in model.Linear.parameters():
        param.requires_grad = True

    # Optimizer
    # Vision Transformers (ViTs) use the AdamW optimizer for training.
    # Learning rate is ften around 1e-4 to 1e-3.
    lora_params = [param for name, param in model.named_parameters() if param.requires_grad and "lora" in name]
    base_params = [param for name, param in model.named_parameters() if param.requires_grad and "lora" not in name]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": 3e-4}, {"params": base_params, "lr": config.lr}], weight_decay=config.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max)
    # Recoder the weights that is trained
    train_parameter_name = []
    for name, param in model.named_parameters():
        # print(name, param.requires_grad)
        if param.requires_grad == True:
            train_parameter_name.append(name)

    lora_total_params = sum(p.numel() for p in lora.lora_state_dict(model).values())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters (only LoRA):", lora_total_params)
    print("Trainable parameters (model_trainable_params):", model_trainable_params)

    trainable_weights = [
        name for name, param in model.named_parameters() if param.requires_grad == True
    ]
    # print("Trainable parameters:", trainable_weights)
    for epoch in range(config.epochs):
        ################ Train ################
        # Set Trainable parameters
        lora.mark_only_lora_as_trainable(model)

        # Unfreeze the Projection layer
        for param in model.Linear.parameters():
            param.requires_grad = True

        train_loss = 0
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch")

        for batch in progress_bar:
            optimizer.zero_grad()
            batch["images"]             = batch["images"].to(device)
            batch["input_ids"]          = batch["input_ids"].to(device)
            batch["attention_masks"]    = batch["attention_masks"].to(device)
            loss = model(   batch["images"],
                            batch["input_ids"],
                            batch["attention_masks"])
            loss.backward()
            optimizer.step()
            # Track loss for the epoch
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        scheduler.step()
        print(f"Epoch {epoch} Train Loss: {train_loss / len(train_loader)}")
        # Save model        
        checkpoint_path = os.path.join(config.output_checkpoint,f"epoch_{epoch}.bin")
        checkpoint = {
            "lora_state_dict": lora.lora_state_dict(model),
            "trainable_params": model.Linear.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        ################ Evaluation ################
        model.eval()
        # Load
        # checkpoint = torch.load(checkpoint_path)
        # lora_params = checkpoint["lora_state_dict"]
        # model.load_state_dict(lora_params, strict=False)
        # model.Linear.load_state_dict(checkpoint["trainable_params"])


        val_loss = 0
        output_data = {}
        model.eval()
        for batch in tqdm(valid_loader):
            batch["images"]     = batch["images"].to(device)
            batch["input_ids"]  = batch["input_ids"].to(device)
            batch["attention_masks"] = batch["attention_masks"].to(device)
            output_ids          = model.generate(batch["images"])
            sentence            = tokenizer.decode(output_ids)

            with torch.no_grad():
                loss = model(   batch["images"],
                                batch["input_ids"],
                                batch["attention_masks"])

            val_loss += loss.item()
            for i in range(len(batch["filenames"])):
                filename = batch["filenames"][i]
                output_data[filename] = sentence

        print(f"Epoch {epoch} Validation Loss: {val_loss / len(valid_loader)}")
        # Save predictions to json
        file_name   = f"Epoch_{epoch}.json"
        path        = os.path.join(config.pred_file,file_name)
        with open(path, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    torch.manual_seed(42)
    main()