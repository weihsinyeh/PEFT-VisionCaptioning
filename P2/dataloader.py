import torch, json, os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

def pad_sequences(sequences, groud_truth, pad_token_id = -100):
    length = []
    for seq in sequences:
        length.append(len(seq))
    max_len = max(length)
    padded_sequences, GT_sequences = [], []
    for seq in sequences:
        padded_sequences.append(seq + [pad_token_id] * (max_len - len(seq)))
    for ground_truth in groud_truth:
        GT_sequences.append(ground_truth + [pad_token_id] * (max_len - len(ground_truth)))
    return padded_sequences, GT_sequences

class DataLoaderTrain(Dataset):
    def __init__(self, imagedir, annotation_json, tokenizer, transform):
        self.imagedir = imagedir
        with open(annotation_json) as f:
            self.annotation = json.load(f)
        self.tokenizer = tokenizer
        self.transform = transform
        self.datas = {}
        dicts = {}
        for data in self.annotation["images"] :
            dicts[data["id"]] = data["file_name"]
        for data in self.annotation["annotations"]:
            data["caption"]         = data["caption"]
            data["token"]           = self.tokenizer.encode(data["caption"])
            data["image_id"]        = data["image_id"]
            data["file_name"]       = dicts[data["image_id"]]
            # transform image only when training
            path                    = os.path.join(self.imagedir, data["file_name"])
            image                   = Image.open(path)
            data["image"]           = self.transform(image).convert('RGB')
            self.datas[data["image_id"]]  = data
    
    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        filenames, captions, input_ids, images, image_ids  = [], [], [], [], []
        for item in batch:
            filenames.append(item["file_name"])
            captions.append(item["caption"])
            input_ids.append(item["token"])
            image_ids.append(item["image_id"])
            images.append(item["image"])

        GT_ids = input_ids[:,1:]
        input_ids, GT_ids = pad_sequences(input_ids, GT_ids)
        attention_masks     = [[float(1) if i != -100 else float(0) for i in input_id] for input_id in input_ids]
        input_ids           = torch.tensor(input_ids)
        GT_ids              = torch.tensor(GT_ids)
        att_mask_tensors    = torch.tensor(attention_masks)
        images              = torch.stack(images, dim = 0)
        return {    filenames       : filenames,
                    captions        : captions,
                    input_ids       : input_ids,
                    GT_ids          : GT_ids,
                    attention_masks : att_mask_tensors,
                    images          : images,
                    image_ids       : image_ids }
    
class DataLoaderTest:
    def __init__(self, imagedir, annotation_json, transform):
        self.imagedir = imagedir
        with open(annotation_json) as f:
            self.annotation = json.load(f)

        self.transform = transform
        self.datas = {}
        for data in self.annotation["images"] :
            data["image"]       = Image.open(self.imagedir / data["file_name"]).convert('RGB')
            data["file_name"]   = data["file_name"]
            data["image_id"]    = data["id"]

            self.datas[data["image_id"]] = data
    
    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        filenames   = []
        images      = []
        image_ids   = []
   
        for item in batch:
            filenames.append(item["file_name"])
            image_ids.append(item["image_id"])
            images.append(item["image"])
            
        return {    filenames: filenames,
                    images: images,
                    image_ids: image_ids }