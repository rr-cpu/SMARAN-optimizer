import torch
from torchvision import datasets, transforms
from pathlib import Path

class CIFAR10:
    def __init__(self, batch_size=64):
        """Initialize the CIFAR-10 dataset with transformations."""
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        self._load_data()
        

    def _load_data(self):
        """Load train and test datasets."""
        self.train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def get_loaders(self):
        """Return train and test data loaders."""
        return self.train_loader, self.test_loader


class CIFAR100:
    def __init__(self, batch_size=64):
        """Initialize the CIFAR-100 dataset with transformations."""
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 stats
        ])
        self._load_data()

    def _load_data(self):
        """Load train and test datasets."""
        self.train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def get_loaders(self):
        """Return train and test data loaders."""
        return self.train_loader, self.test_loader


class Mnist:
    def __init__(self, batch_size=64):
        """Initialize the MNIST dataset with transformations."""
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self._load_data()

    def _load_data(self):
        """Load train and test datasets."""
        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1000, shuffle=False, pin_memory=True)

    def get_loaders(self):
        """Return train and test data loaders."""
        return self.train_loader, self.test_loader




class TinyImageNetLoader:
    def __init__(self, root_dir, batch_size=128, num_workers=4, image_size=64):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_dir = self.root_dir / "train"
        self.val_dir = self.root_dir / "val/structured"
        self.train_tf, self.val_tf = self._build_transforms()
        self.train_loader,self.test_loader= self.get_loaders()

    def _build_transforms(self):
        mean = [0.480, 0.448, 0.398]
        std = [0.277, 0.269, 0.282]
        train_tf = transforms.Compose([
            transforms.RandomCrop(self.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train_tf, val_tf

    def get_loaders(self):
        train_ds = datasets.ImageFolder(self.train_dir, transform=self.train_tf)
        val_ds = datasets.ImageFolder(self.val_dir, transform=self.val_tf)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return train_loader, val_loader



        
