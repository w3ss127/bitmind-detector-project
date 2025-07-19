import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import os
import timm

# ---------------------------------
# 1. Model Definition
# ---------------------------------
transforms = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes, resnet_variant="resnet50"):
        super().__init__()
        if resnet_variant == "resnet50":
            self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        elif resnet_variant == "resnet121":
            self.resnet = models.resnet121(weights='IMAGENET1K_V2')
        else:
            raise ValueError("resnet_variant must be 'resnet50' or 'resnet121'")
        resnet_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_fc_in = self.vit.head.in_features
        self.vit.head = nn.Identity()

        self.fc1 = nn.Linear(resnet_fc_in + vit_fc_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        res_feat = self.resnet(x)
        vit_feat = self.vit(x)
        combined = torch.cat([res_feat, vit_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

# ---------------------------------
# 2. Detector Class
# ---------------------------------
class Detector:
    def __init__(self, config, model_path=None, device=None):
        if model_path is not None and not isinstance(model_path, str):
            raise TypeError(f"model_path must be a string or None, got {type(model_path)}")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "best_model.pt")
        assert os.path.exists(self.model_path), f"❌ Model file not found: {self.model_path}"

        self.model = ResNetViTHybrid(num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Loaded model from {self.model_path}")

    def detect(self, media_tensor: torch.Tensor, modality: str = "image"):
        if modality == "image":
            media_tensor = transforms(media_tensor)
            print(f"IMAGE: {media_tensor}")
            media_tensor = media_tensor.to(self.device)  # Move tensor to the same device as model
            # Add batch dimension if not present
            if media_tensor.dim() == 3:
                media_tensor = media_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            with torch.no_grad():
                probs = F.softmax(self.model(media_tensor), dim=1)
            return probs.squeeze().cpu().tolist()

        elif modality == "video":
            # Assume input is [C, T, H, W] or [T, C, H, W]
            # We'll handle [C, T, H, W] (PyTorch convention)
            print(f"VIDEO: {media_tensor}")
            if media_tensor.dim() == 4 and media_tensor.shape[0] in [1, 3]:
                # [C, T, H, W] -> [T, C, H, W]
                media_tensor = media_tensor.permute(1, 0, 2, 3)
            frame_probs = []
            with torch.no_grad():
                for frame in media_tensor:  # frame: [C, H, W]
                    frame = transforms(frame)
                    frame = frame.to(self.device)
                    if frame.dim() == 3:
                        frame = frame.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                    probs = F.softmax(self.model(frame), dim=1)
                    frame_probs.append(probs.squeeze(0))
            avg_probs = torch.stack(frame_probs).mean(dim=0)
            return avg_probs.cpu().tolist()

        else:
            raise ValueError("Modality must be 'image' or 'video'")


