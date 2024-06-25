import torch
import torchvision.transforms as transforms

from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader


class CocoDetectionWithFilenames(CocoDetection):
    def __init__(self, root: str, ann_file: str, transform=None):
        super().__init__(root, ann_file, transform)

    def get_filename(self, idx: int) -> str:
        return self.coco.loadImgs(self.ids[idx])[0]["file_name"]


def get_loaders(
        root: str,
        ann_file: str,
        num_workers: int = 2,
        prefetch_factor: int = 2
) -> tuple[CocoDetection, DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CocoDetectionWithFilenames(
        root=root,
        ann_file=ann_file,
        transform=transform
    )
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
    )
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataset, train_loader, valid_loader, test_loader
