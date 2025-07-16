import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

# ---------------------------------
# 1. Model Definition
# ---------------------------------
class ResNetDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone_out_channels = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
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
        return self.classifier(x)

# ---------------------------------
# 2. Detector Class
# ---------------------------------
class Detector:
    def __init__(self, config, model_path=None, device=None):
        if model_path is not None and not isinstance(model_path, str):
            raise TypeError(f"model_path must be a string or None, got {type(model_path)}")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "bitmind.pth")
        assert os.path.exists(self.model_path), f"❌ Model file not found: {self.model_path}"

        self.model = ResNetDeepfakeDetector(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Loaded model from {self.model_path}")

    def _preprocess_image(self, img_tensor):
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
        img_tensor = img_tensor.float() / 255.0
        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return img_tensor.to(self.device)

    def _preprocess_video(self, video_tensor):
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t, :, :]  # [C, H, W]
            frame = self._preprocess_image(frame)
            frames.append(frame)
        return torch.cat(frames, dim=0)  # [T, 3, 224, 224]

    def detect(self, media_tensor: torch.Tensor, modality: str = "image"):
        if modality == "image":
            input_tensor = self._preprocess_image(media_tensor)
            with torch.no_grad():
                probs = F.softmax(self.model(input_tensor), dim=1)
            return probs.squeeze().cpu().tolist()

        elif modality == "video":
            input_tensor = self._preprocess_video(media_tensor)
            frame_probs = []
            with torch.no_grad():
                for frame in input_tensor:
                    frame = frame.unsqueeze(0)  # [1, 3, 224, 224]
                    probs = F.softmax(self.model(frame), dim=1)
                    frame_probs.append(probs.squeeze(0))
            avg_probs = torch.stack(frame_probs).mean(dim=0)
            return avg_probs.cpu().tolist()

        else:
            raise ValueError("Modality must be 'image' or 'video'")


