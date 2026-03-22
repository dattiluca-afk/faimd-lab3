import os
import shutil
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

def prepare_data():
    """Downloads and reformats Tiny-ImageNet using native Python."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_name = "tiny-imagenet-200.zip"
    extract_path = "data" 

    # 1. Download
    if not os.path.exists(zip_name):
        print("Downloading dataset (120MB)...")
        urllib.request.urlretrieve(url, zip_name)
    
    # 2. Unzip
    # We check if the 'train' folder exists to avoid unzipping every time
    if not os.path.exists(os.path.join(extract_path, 'tiny-imagenet-200', 'train')):
        print(f"Unzipping into {extract_path} folder...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # 3. Reformat Validation Folder
    val_dir = os.path.join(extract_path, 'tiny-imagenet-200', 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    
    if os.path.exists(val_images_dir):
        print("Reformatting validation directory structure...")
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

def get_dataloaders(batch_size=32, num_workers=4):
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

    # We use absolute paths to be 100% sure where we are looking
    cwd = os.getcwd()
    base_path = os.path.join(cwd, 'data', 'tiny-imagenet-200')
    train_root = os.path.join(base_path, 'train')
    val_root = os.path.join(base_path, 'val')

    print(f"Checking for data in: {train_root}")

    if not os.path.exists(train_root):
        # Let's see what IS in the data folder to help debug
        if os.path.exists(os.path.join(cwd, 'data')):
            content = os.listdir(os.path.join(cwd, 'data'))
            print(f"Content of 'data' folder: {content}")
        raise FileNotFoundError(f"CRITICAL: Could not find training folder at {train_root}")

    train_set = ImageFolder(root=train_root, transform=train_transform)
    val_set = ImageFolder(root=val_root, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader