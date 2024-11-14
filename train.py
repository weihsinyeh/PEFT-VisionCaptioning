import timm, argparse, torch
from tokenizer import BPETokenizer
from P2.dataloader import DataLoaderTrain
from P2.transform import augmentation
from P2.model import VITModel
from decoder import Decoder, Config

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file",    type = str, default = "./hw3_data/p2_data/train.json")
    parser.add_argument("--pred_file",          type = str, default = "./P2_output/predictions.json")
    parser.add_argument("--images_dir",         type = str, default = "./hw3_data/p2_data/images/train")
    parser.add_argument("--decoder",            type = str, default = "./hw3_data/p2_data/decoder_model.bin")
    parser.add_argument("--batch_size",         type = int, default = 8)
    parser.add_argument("--lr",                 type = float, default = 1e-4)
    parser.add_argument("--epochs",             type = int, default = 100)
    return parser.parse_args()

def main():
    config = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load Tokenizer
    Tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
    # Load Dataset
    TrainDataset = DataLoaderTrain(config.images_dir, config.annotation_file, Tokenizer, augmentation)
    
    # Load Encoder : ViT-Large from timm
    pretrained_model = timm.create_model('vit_large_patch16_224', pretrained=True).to(device)
    
    # Load Decoder
    Decoder = Decoder(Config(config.decoder)).to(device)
    
    # model = VITModel(pretrained_model, Decoder, Tokenizer).to(device)


if __name__ == '__main__':
    torch.manual_seed(42)
    main()