import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import timm

class ResNetViTHybrid(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.fc1 = nn.Linear(self.resnet.num_features + self.vit.num_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resnet_features = self.resnet(x)
        vit_features = self.vit(x)
        combined = torch.cat((resnet_features, vit_features), dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

class Detector:

    def __init__(self, model_path: str = None, device: str | None = None):
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set model path
        self.model_path = os.path.join(os.path.dirname(__file__), "best_model_60000_2.pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize model and load weights
        self.model = ResNetViTHybrid(num_classes=3).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        print(f"Loaded model from {self.model_path}")

        # Define preprocessing pipeline
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),  # Normalize if in [0, 255]
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect(self, media_tensor: torch.Tensor, modality: str = "image") -> list[float]:
        media_tensor = media_tensor.to(self.device)  # Move to device
        valid_modalities = ["image", "video"]

        if modality not in valid_modalities:
            raise ValueError(f"Modality must be one of {valid_modalities}, got '{modality}'")

        if modality == "image":
            # Validate image tensor shape
            if media_tensor.dim() not in [3, 4]:
                raise ValueError(f"Expected image tensor of shape [C, H, W] or [1, C, H, W], got shape {media_tensor.shape}")
            if media_tensor.dim() == 4:
                if media_tensor.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1 for image tensor, got shape {media_tensor.shape}")
                media_tensor = media_tensor.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            if media_tensor.shape[0] not in [1, 3]:
                raise ValueError(f"Expected 1 or 3 channels for image tensor, got shape {media_tensor.shape}")

            # Apply preprocessing
            try:
                media_tensor = self.transforms(media_tensor)
            except (RuntimeError, TypeError) as e:
                raise ValueError(f"Error preprocessing image: {e}")

            media_tensor = media_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            with torch.no_grad():
                self.model.eval()  # Set evaluation mode
                probs = F.softmax(self.model(media_tensor), dim=1)
            return probs.squeeze().cpu().tolist()

        # Video modality
        if media_tensor.dim() != 4:
            raise ValueError(f"Expected video tensor of shape [C, T, H, W] or [T, C, H, W], got shape {media_tensor.shape}")

        # Handle [C, T, H, W] or [T, C, H, W]
        if media_tensor.shape[0] in [1, 3]:
            media_tensor = media_tensor.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
        elif media_tensor.shape[1] not in [1, 3]:
            raise ValueError(f"Expected video tensor with 1 or 3 channels, got shape {media_tensor.shape}")

        frame_probs = []
        with torch.no_grad():
            self.model.eval()  # Set evaluation mode
            for frame in media_tensor:  # frame: [C, H, W]
                if frame.shape[0] not in [1, 3]:
                    raise ValueError(f"Expected 1 or 3 channels for video frame, got shape {frame.shape}")
                try:
                    frame = self.transforms(frame)
                except (RuntimeError, TypeError) as e:
                    raise ValueError(f"Error preprocessing video frame: {e}")
                frame = frame.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                probs = F.softmax(self.model(frame), dim=1)
                frame_probs.append(probs.squeeze(0))
        avg_probs = torch.stack(frame_probs).mean(dim=0)
        return avg_probs.cpu().tolist()