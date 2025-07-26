import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import bittensor as bt
import os
from torchvision import transforms, models
from PIL import Image

class ResNetDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3, model_name='resnet50'):
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

class Detector:
    """Handler for image and video detection models."""
    def __init__(self, config, model_path=None):
        bt.logging(config=config, logging_dir=config.neuron.full_path)
        bt.logging.set_info()
        if config.logging.debug:
            bt.logging.set_debug(True)
        if config.logging.trace:
            bt.logging.set_trace(True)

        self.config = config
        self.image_detector = None
        self.video_detector = None
        self.device = (
            self.config.device
            if hasattr(self.config, "device")
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Allow custom model path
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(os.path.dirname(__file__), "bitmind.pth")

    def load_model(self, modality=None):
        bt.logging.info(f"Loading {modality} detection model from: {self.model_path}")

        assert os.path.exists(self.model_path), "❌ Model weights not found!"
    
        model = ResNetDeepfakeDetector(num_classes=3).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()

        if modality in ("image", None):
            try:
                self.image_detector = model
                bt.logging.info("✅ Image detector loaded successfully!")
            except Exception as e:
                bt.logging.error(f"❌ Error loading image model: {e}")
                raise
        if modality in ("video", None):
            try:
                self.video_detector = model
                bt.logging.info("✅ Video detector loaded successfully!")
            except Exception as e:
                bt.logging.error(f"❌ Error loading video model: {e}")
                raise

    def preprocess_image(self, image_tensor):
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 4 and image_tensor.shape[1] == 3:
            pass  # already [1, 3, H, W]
        else:
            raise ValueError(f"Unsupported image shape: {image_tensor.shape}")

        image_tensor = image_tensor.float() / 255.0
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        bt.logging.error("image_tensor.shape:",image_tensor.shape)
        return image_tensor.to(self.device)


    def preprocess_video(self, video_tensor):
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t, :, :]
            processed_frame = self.preprocess_image(frame)
            frames.append(processed_frame)
        return torch.cat(frames, dim=0)

    def preprocess(self, media_tensor, modality):
        bt.logging.debug({
            "modality": modality,
            "shape": tuple(media_tensor.shape),
            "dtype": str(media_tensor.dtype),
            "min": torch.min(media_tensor).item(),
            "max": torch.max(media_tensor).item(),
        })
        if modality == "image":
            return self.preprocess_image(media_tensor)
        elif modality == "video":
            return self.preprocess_video(media_tensor)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def detect(self, media_tensor, modality):
        media_tensor = self.preprocess(media_tensor, modality)
        if modality == "image":
            if self.image_detector is None:
                self.load_model("image")
                
            if self.image_detector is None:
                raise RuntimeError("Image detector model is not loaded.")
            
            
        elif modality == "video":
            if self.video_detector is None:
                self.load_model("video")
            if self.video_detector is None:
                raise RuntimeError("Video detector model is not loaded.")
            frame_predictions = []
            with torch.no_grad():
                for i in range(media_tensor.shape[0]):
                    frame = media_tensor[i:i+1]
                    outputs = self.video_detector(frame)
                    probs = F.softmax(outputs, dim=1)
                    frame_predictions.append(probs[0])
                probs = torch.stack(frame_predictions).mean(dim=0)
            return probs
        else:
            raise ValueError(f"Unsupported modality: {modality}")