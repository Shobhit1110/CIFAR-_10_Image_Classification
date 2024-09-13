from torchvision import datasets, transforms
import torch.utils.data

def load_data_cifar10(batch_size):
    """Download the CIFAR-10 dataset and then load it into memory with data augmentation and normalization."""
    train_trans = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]
    test_trans = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]
    train_transform = transforms.Compose(train_trans)
    test_transform = transforms.Compose(test_trans)
    cifar10_train = datasets.CIFAR10(root="../data", train=True, transform=train_transform, download=True)
    cifar10_test = datasets.CIFAR10(root="../data", train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
