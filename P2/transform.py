from torchvision import transforms

augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
    ])