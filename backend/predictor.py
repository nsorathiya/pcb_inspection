import os
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class PCBClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PCBClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        conv1_out = x.clone()
        x = self.pool(F.relu(self.conv2(x)))
        conv2_out = x.clone()
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        fc1_out = x.clone()
        x = self.fc2(x)
        return x, {'conv1': conv1_out, 'conv2': conv2_out, 'fc1': fc1_out}

class Predictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PCBClassifier(num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.labels = ["missing_part", "dispense_error", "misalignment", "no_defect"]

    def predict_image(self, img_path):
        img = cv2.imread(img_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output, activations = self.model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, 1).item()
            label = self.labels[pred_class]
            confidence = probs[0, pred_class].item()

        # 🔴 Draw circle + label on predicted defective image
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        if label != "no_defect":
            cv2.circle(img, (center_x, center_y), 40, (0, 0, 255), 2)
            cv2.putText(img, f"{label} ({confidence:.2f})", (center_x - 80, center_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🧠 Save neuron activation tensors
        os.makedirs("neuron_data", exist_ok=True)
        for layer, activation in activations.items():
            layer_path = os.path.join("neuron_data", f"{layer}.pt")
            torch.save(activation.cpu(), layer_path)

        return label, img
