from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from app.core.class_labels import load_class_label_contract
from app.core.model_compatibility import write_verified_model_label_metadata
from data_loader import get_dataloaders
from model import PCBClassifier


class Trainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        loader, class_to_idx = get_dataloaders(self.data_dir)
        contract = load_class_label_contract()
        contract.validate_class_to_idx(class_to_idx, source="Training dataset")
        model = PCBClassifier(num_classes=contract.class_count).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        model_path = Path("saved_model") / "best_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        write_verified_model_label_metadata(model_path, contract, class_to_idx)
