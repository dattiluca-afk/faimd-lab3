import os
import shutil
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

def prepare_data():
    """Downloads and reformats Tiny-ImageNet using native Python (Windows friendly)."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_name = "tiny-imagenet-200.zip"
    extract_path = "/data"

    # 1. Download
    if not os.path.exists(zip_name):
        print("Downloading dataset (this may take a minute)...")
        urllib.request.urlretrieve(url, zip_name)
    
    # 2. Unzip
    if not os.path.exists(extract_path):
        print("Unzipping...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # 3. Reformat Validation Folder
    # Using os.path.join makes it work correctly on Windows
    val_dir = os.path.join(extract_path, 'tiny-imagenet-200', 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    
    if os.path.exists(val_images_dir):
        print("Reformatting validation directory...")
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            for line in f:
                parts = line.split('\t')
                fn, cls = parts[0], parts[1]
                
                target_class_dir = os.path.join(val_dir, cls)
                os.makedirs(target_class_dir, exist_ok=True)
                
                src_path = os.path.join(val_images_dir, fn)
                dst_path = os.path.join(target_class_dir, fn)
                if os.path.exists(src_path):
                    shutil.copyfile(src_path, dst_path)
                    
        shutil.rmtree(val_images_dir)
        print("Data preparation complete!")

def get_dataloaders(batch_size=32, num_workers=0):
    # Note: num_workers=0 is recommended for Windows to avoid Multiprocessing errors
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Path to data
    train_root = os.path.join('tiny-imagenet', 'tiny-imagenet-200', 'train')
    val_root = os.path.join('tiny-imagenet', 'tiny-imagenet-200', 'val')

    train_set = ImageFolder(root=train_root, transform=train_transform)
    val_set = ImageFolder(root=val_root, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader