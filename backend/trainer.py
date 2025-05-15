import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import get_dataloaders
from model import PCBClassifier

class Trainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        loader, class_to_idx = get_dataloaders(self.data_dir)
        model = PCBClassifier(num_classes=len(class_to_idx)).to(self.device)
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

        os.makedirs("saved_model", exist_ok=True)
        torch.save(model.state_dict(), "saved_model/best_model.pth")
