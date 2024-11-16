import torch, json, os
from PIL import Image
from torch.utils.data import Dataset

PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS = 50256

# GT and input ids is padded to the same length and pad_token_id is (-1)
def pad_sequences(sequences, pad_token_id = -1):
    length = []
    for seq in sequences:
        length.append(len(seq))
    max_len = max(length)
    padded_sequences = []
    for seq in sequences:
        padded_sequences.append(seq + [pad_token_id] * (max_len - len(seq)))
    return padded_sequences

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
            data["image_id"]        = data["image_id"]
            data["file_name"]       = dicts[data["image_id"]]
            # transform image only when training
            self.datas[data["image_id"]]  = data
    
    def __getitem__(self, idx):
        # transform image only when training
        path = os.path.join(self.imagedir, self.datas[idx]["file_name"])
        image = Image.open(path).convert('RGB')
        self.datas[idx]["image"] = self.transform(image)
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        filenames, captions, input_ids, images, image_ids  = [], [], [], [], []
        for item in batch:
            filenames.append(item["file_name"])
            captions.append(item["caption"])
            image_ids.append(item["image_id"])
            images.append(item["image"])

        input_ids = []
        for caption in captions:
            input_id = self.tokenizer.encode(caption)
            # BOS : 50256
            if input_id[0] != BOS:
                input_id.insert(0, BOS)
            if input_id[-1] != BOS:
                input_id.insert(len(input_id), BOS)
            input_ids.append(input_id)

        input_ids           = pad_sequences(input_ids, -1)
        attention_masks     = [[float(i != -1) for i in input_id] for input_id in input_ids]

        input_ids           = [[PAD_TOKEN if x == -1 else x for x in seq] for seq in input_ids]

        input_ids           = torch.tensor(input_ids)

        att_mask_tensors    = torch.tensor(attention_masks)

        images = torch.stack(images, dim=0)
        return {    "filenames"       : filenames,
                    "captions"        : captions,
                    "input_ids"       : input_ids,
                    "attention_masks" : att_mask_tensors,
                    "images"          : images,
                    "image_ids"       : image_ids }
    
class DataLoaderTest(Dataset):
    def __init__(self, imagedir, image2tensor):
        self.imagedir = imagedir
        self.images = os.listdir(imagedir)
        self.image2tensor = image2tensor
    
    def __getitem__(self, idx):
        img = self.image2tensor(Image.open(os.path.join(self.imagedir, self.images[idx])).convert('RGB'))
        return {
            "image": img,
            "file_name": self.images[idx][0]}

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        filenames, images   = [], []
        print("batch", batch)
        for item in batch:
            filenames.append(item["file_name"])
            images.append(item["image"])
        images = torch.stack(images, dim=0)
        return {    "filenames": filenames,
                    "images": images}