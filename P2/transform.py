from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch
from PIL import Image
import torchvision.transforms as transforms
augmentation = create_transform(**resolve_data_config({}, model='vit_base_patch16_224.augreg_in21k'))