from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from P2.setting import modelname

augmentation    = create_transform(**resolve_data_config({}, model=modelname))
config      = resolve_data_config({}, model=modelname)
mean        = config['mean']
std         = config['std']
input_size  = config['input_size'][1:]
transform   = Compose([
    Resize(input_size),
    ToTensor(),
    Normalize(mean=mean, std=std)
])