import torch, json, os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class DataLoaderTrain(Dataset):
    def __init__(self, imagedir, annotation_json):
        self.imagedir = imagedir
        with open(annotation_json) as f:
            self.annotation = json.load(f)

        self.datas = {}
        self.transform = transforms.ToTensor()
        dicts = {}
        for data in self.annotation["images"] :
            dicts[data["id"]] = data["file_name"]
        for data in self.annotation["annotations"]:
            item = {}
            item["caption"]         = data["caption"]
            item["image_id"]        = data["image_id"]
            file_name, ext = os.path.splitext(dicts[data["image_id"]])
            item["file_path"]       = dicts[data["image_id"]]
            item["file_name"]       = file_name
            # transform image only when training
            self.datas[data["image_id"]]  = item
    
    def __getitem__(self, idx):
        # transform image only when training
        path = os.path.join(self.imagedir, self.datas[idx]["file_path"])
        image = Image.open(path).convert('RGB')
        self.datas[idx]["image"] = image
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

        return {    "filenames"       : filenames,
                    "captions"        : captions,
                    "images"          : images,
                    "image_ids"       : image_ids }
    
class DataLoaderTest(Dataset):
    def __init__(self, imagedir):
        self.imagedir = imagedir
        self.images = os.listdir(imagedir)
        self.transform = transforms.ToTensor()
        self.datas = []
        for data in self.images:
            item = {}
            filename, ext = os.path.splitext(data)
            item["filepath"] = data
            item["filename"] = filename
            self.datas.append(item)
        
    def __getitem__(self, idx):
        path = os.path.join(self.imagedir, self.datas[idx]["filepath"])
        image = Image.open(path).convert('RGB')
        self.datas[idx]["image"] = image
        return self.datas[idx]

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        filenames, images, filepaths = [], [], []
        for item in batch:
            filenames.append(item["filename"])
            images.append(item["image"])
            filepaths.append(item["filepath"])
        return {    "filenames" : filenames,
                    "images"    : images,
                    "filepaths" : filepaths}