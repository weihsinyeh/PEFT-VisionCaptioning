import torch, json, os
from PIL import Image
from torch.utils.data import Dataset

class DataLoaderTrain(Dataset):
    def __init__(self, imagedir, annotation_json, tokenizer):
        self.imagedir = imagedir
        with open(annotation_json) as f:
            self.annotation = json.load(f)
        self.tokenizer = tokenizer
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
        image = Image.open(path).convert('RGB').toTensor()
        self.datas[idx]["image"] = self.transform(image)
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)

    def collate_fn(self, batch):
        filenames, captions, images, image_ids  = [], [], [], []
        for item in batch:
            filenames.append(item["file_name"])
            captions.append(item["caption"])
            image_ids.append(item["image_id"])
            images.append(item["image"])

        images = torch.stack(images, dim=0)
        return {    "filenames"       : filenames,
                    "captions"        : captions,
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