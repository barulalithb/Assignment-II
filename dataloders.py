import torchvision
import torchvision.transforms as tt
from torch.utils.data import DataLoader


def load_neuro_face(opt):
    train_data_path = "./data/train"
    test_data_path = "./data/test"

    train_stats = ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    test_stats = ((0.4824, 0.4495, 0.3981), (0.2301, 0.2264, 0.2261))
    policies = tt.AutoAugmentPolicy.IMAGENET

    train_transform = tt.Compose(
        [
            tt.RandomCrop(64, padding=4),
            tt.RandomHorizontalFlip(),
            # tt.AutoAugment(policies),
            # tt.RandAugment(num_ops=2, magnitude=7),
            tt.ToTensor(),
            tt.Normalize(*train_stats),
        ]
    )

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*test_stats)])

    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path, transform=train_transform
    )
    train_data_loader = DataLoader(
        train_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = DataLoader(
        test_data,
        opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    return train_data_loader, test_data_loader
