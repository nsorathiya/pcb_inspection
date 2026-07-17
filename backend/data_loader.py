from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.core.class_labels import load_class_label_contract


def get_dataloaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    contract = load_class_label_contract()
    contract.validate_class_to_idx(
        dataset.class_to_idx,
        source=f"ImageFolder dataset {data_dir}",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.class_to_idx
