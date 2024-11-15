from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
augmentation = create_transform(**resolve_data_config({}, model="vit_huge_patch14_clip_224.laion2b"))