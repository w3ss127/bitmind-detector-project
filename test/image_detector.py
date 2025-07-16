import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

# Custom model with classifier
class ResNetDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # --- Define ResNetCustomTop at the top level so torch.load can find it ---
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone_out_channels = backbone.fc.in_features
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # remove last FC and pool

        self.backbone = backbone
        self.norm = nn.BatchNorm2d(backbone_out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_names():
    return ["real", "synthetic", "semi-synthetic"]

def load_model(checkpoint_path, device):
    assert os.path.exists(checkpoint_path), "❌ Model weights not found!"

    model = ResNetDeepfakeDetector(num_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_and_preprocess_images(image_path, device):
    images = torch.load(image_path, map_location=device)
    fixed = []
    converted, skipped = 0, 0
    for i, img in enumerate(images):
        if img.shape == (1, 256, 256):
            img = img.repeat(3, 1, 1)
            converted += 1
        if img.shape != (3, 256, 256):
            print(f"❌ Skipping image {i} with shape {img.shape}")
            skipped += 1
            continue
        fixed.append(img)
    print(f"✅ Fixed: {len(fixed)} | Grayscale converted: {converted} | Skipped: {skipped}")
    return torch.stack(fixed)

def preprocess_image(img, device):
    img = img.unsqueeze(0).float() / 255.0  # [1, 3, 256, 256]
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    return img.to(device)

def model_evaluate(model, images, class_names, device):
    total = 0
    correct = 0
    print(images.shape)
    print(images[0].dtype)
    for idx, img in enumerate(images):
        input_tensor = preprocess_image(img, device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            print(f"Output {idx}, {pred}: {probs[0]}")
            pred_label = class_names[pred]
            true_label = "real"
            print(f"Image {idx}: Predicted={pred_label}, Actual={true_label}")
            if pred_label == true_label:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"\nModel accuracy for {true_label} class: {accuracy*100:.2f}% ({correct}/{total})")

def main():
    device = get_device()
    class_names = get_class_names()
    # Paths
    test_data_path = os.path.join(os.path.dirname(__file__), 'bm_real_images.pt')
    model_path = os.path.join(os.path.dirname(__file__), '../neurons/miner/bitmind.pth')
    # Load and preprocess images
    images = load_and_preprocess_images(test_data_path, device)
    print(f"Loaded {len(images)} images for inference.")
    # Load model
    model = load_model(model_path, device)
    # Evaluate
    model_evaluate(model, images, class_names, device)

if __name__ == "__main__":
    main()