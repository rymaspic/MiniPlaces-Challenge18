import torch
from torchvision import datasets, transforms

data_root = './data/'
train_root = data_root + 'train'
val_root = data_root + 'val'
test_root = data_root + 'test'

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
test_dataset = datasets.ImageFolder(root=test_root, transform=base_transform)

def get_data_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (train_loader, val_loader)

def get_val_test_loaders(batch_size):
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (val_loader, test_loader)
