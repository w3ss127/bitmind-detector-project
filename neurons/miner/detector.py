import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, convnext_small
import timm
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SuperiorConfig:
    def __init__(self):
        self.MODEL_TYPE = "superior_forensics_model"
        self.CONVNEXT_BACKBONE = "convnext_tiny"
        self.PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
        self.NUM_CLASSES = 3
        self.HIDDEN_DIM = 1536
        self.DROPOUT_RATE = 0.4
        self.FREEZE_BACKBONES = True
        self.ATTENTION_DROPOUT = 0.2
        self.USE_SPECTRAL_NORM = True
        self.USE_FORENSICS_MODULE = True
        self.USE_UNCERTAINTY_ESTIMATION = True
        self.IMAGE_SIZE = 224
        self.CLASS_NAMES = ["real", "semi-synthetic", "synthetic"]

class ForensicsAwareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.noise_analyzer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.edge_analyzer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.forensics_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 96, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
    
    def extract_edge_inconsistencies(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge_feats = self.edge_analyzer(gray)
        return edge_feats
    
    def forward(self, x):
        dct_feats = self.dct_analyzer(x)
        noise_feats = self.noise_analyzer(x)
        edge_feats = self.extract_edge_inconsistencies(x)
        freq_feats = self.freq_analyzer(x)
        combined_feats = torch.cat([dct_feats, noise_feats, edge_feats, freq_feats], dim=1)
        forensics_output = self.forensics_fusion(combined_feats)
        return forensics_output

class UncertaintyModule(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes),
            nn.Softplus()
        )
        self.aleatoric_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_classes),
            nn.Softplus()
        )
    
    def forward(self, x):
        evidence = self.evidence_layer(x)
        aleatoric = self.aleatoric_layer(x)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        epistemic_uncertainty = self.num_classes / S
        aleatoric_uncertainty = aleatoric
        return probs, epistemic_uncertainty, aleatoric_uncertainty, alpha

class SuperiorAttentionModule(nn.Module):
    def __init__(self, in_features, config):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.forensics_attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=8,
            dropout=config.ATTENTION_DROPOUT,
            batch_first=True
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Dropout(config.ATTENTION_DROPOUT),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        if config.USE_SPECTRAL_NORM:
            self.channel_attention[0] = nn.utils.spectral_norm(self.channel_attention[0])
            self.channel_attention[3] = nn.utils.spectral_norm(self.channel_attention[3])
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() != 2:
            x = x.view(batch_size, -1)
        x_reshaped = x.unsqueeze(1)
        attn_output, _ = self.forensics_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)
        channel_weights = self.channel_attention(x)
        attended_features = x * channel_weights + attn_output
        return attended_features

class SuperiorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.CONVNEXT_BACKBONE == 'convnext_tiny':
            self.convnext = convnext_tiny(weights=config.PRETRAINED_WEIGHTS)
        elif config.CONVNEXT_BACKBONE == 'convnext_small':
            self.convnext = convnext_small(weights=config.PRETRAINED_WEIGHTS)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        convnext_features = 768
        vit_features = self.vit.num_features
        forensics_features = 128 if config.USE_FORENSICS_MODULE else 0
        total_features = convnext_features + vit_features + forensics_features
        
        if config.USE_FORENSICS_MODULE:
            self.forensics_module = ForensicsAwareModule(config)
        
        self.attention_module = SuperiorAttentionModule(total_features, config)
        
        if config.USE_UNCERTAINTY_ESTIMATION:
            self.uncertainty_module = UncertaintyModule(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        self.classifier = nn.Linear(config.HIDDEN_DIM // 4, config.NUM_CLASSES)
        
        if config.USE_SPECTRAL_NORM:
            self.fusion[0] = nn.utils.spectral_norm(self.fusion[0])
            self.fusion[3] = nn.utils.spectral_norm(self.fusion[3])
            self.fusion[6] = nn.utils.spectral_norm(self.fusion[6])
            self.classifier = nn.utils.spectral_norm(self.classifier)
    
    def forward(self, x):
        convnext_feats = self.convnext.features(x)
        convnext_feats = self.convnext.avgpool(convnext_feats)
        convnext_feats = torch.flatten(convnext_feats, 1)
        
        vit_feats = self.vit.forward_features(x)
        vit_feats = vit_feats[:, 0]
        
        features_list = [convnext_feats, vit_feats]
        
        if self.config.USE_FORENSICS_MODULE:
            forensics_feats = self.forensics_module(x)
            features_list.append(forensics_feats)
        
        fused_features = torch.cat(features_list, dim=1)
        attended_features = self.attention_module(fused_features)
        processed_features = self.fusion(attended_features)
        logits = self.classifier(processed_features)
        
        if self.config.USE_UNCERTAINTY_ESTIMATION and hasattr(self, 'uncertainty_module'):
            probs, epistemic_unc, aleatoric_unc, alpha = self.uncertainty_module(processed_features)
            return logits, processed_features, (probs, epistemic_unc, aleatoric_unc, alpha)
        
        return logits, processed_features

class Detector:
    def __init__(self, model_path: str = None, device: str | None = None):
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Look for the best model from training
        self.model_path = os.path.join(os.path.dirname(__file__), "best_superior_model.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Initialize config and model
        self.config = SuperiorConfig()
        self.model = SuperiorModel(self.config).to(self.device)
        
        # Load model weights with PyTorch 2.6 compatibility
        try:
            # First try with weights_only=True (safer, PyTorch 2.6 default)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=True: {e}")
            # Fallback to weights_only=False (for compatibility with older checkpoints)
            print("Falling back to weights_only=False - only use with trusted checkpoints!")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from distributed training)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Handle legacy checkpoint format compatibility
        def remap_legacy_keys(state_dict):
            """Remap keys from legacy checkpoint format to current model structure"""
            key_mapping = {}
            
            for old_key in list(state_dict.keys()):
                new_key = old_key
                
                # Map forensics module keys
                if old_key.startswith("forensics_"):
                    new_key = old_key.replace("forensics_", "forensics_module.", 1)
                
                # Map attention module keys  
                elif old_key.startswith("attention_"):
                    new_key = old_key.replace("attention_", "attention_module.", 1)
                
                # Map uncertainty module keys
                elif old_key.startswith("uncertainty_"):
                    new_key = old_key.replace("uncertainty_", "uncertainty_module.", 1)
                
                if new_key != old_key:
                    key_mapping[old_key] = new_key
            
            # Apply the remapping
            for old_key, new_key in key_mapping.items():
                state_dict[new_key] = state_dict.pop(old_key)
            
            return state_dict
        
        # Check if this is a legacy checkpoint and remap keys
        legacy_keys = any(k.startswith(("forensics_", "attention_", "uncertainty_")) and 
                         not k.startswith(("forensics_module.", "attention_module.", "uncertainty_module.")) 
                         for k in state_dict.keys())
        
        if legacy_keys:
            print("Detected legacy checkpoint format, remapping keys...")
            state_dict = remap_legacy_keys(state_dict)
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded Superior model from {self.model_path}")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"Warning: Strict loading failed, trying non-strict loading: {e}")
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                    print("Loaded model with non-strict mode - some layers may be randomly initialized")
                except Exception as e2:
                    print(f"Error: Could not load model even with non-strict mode: {e2}")
                    raise
            else:
                print(f"Warning: Could not load Superior model state dict: {e}")
                print("This might be an older ResNet model format. Please use a Superior model checkpoint.")
                raise
        
        self.model.eval()
        
        # Define preprocessing pipeline using Albumentations (similar to training)
        self.transform = A.Compose([
            A.Resize(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Fallback torchvision transforms for tensor input
        self.tensor_transforms = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0 if x.max() > 1.0 else x),  # Normalize if in [0, 255]
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _preprocess_tensor(self, tensor):
        """Preprocess a tensor using torchvision transforms"""
        return self.tensor_transforms(tensor)
    
    def _preprocess_numpy(self, image_np):
        """Preprocess a numpy image using albumentations"""
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        transformed = self.transform(image=image_np)
        return transformed['image']
    
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
                # Try to convert to numpy for albumentations
                if media_tensor.requires_grad:
                    media_tensor = media_tensor.detach()
                
                image_np = media_tensor.cpu().permute(1, 2, 0).numpy()
                media_tensor = self._preprocess_numpy(image_np)
            except Exception:
                # Fallback to tensor transforms
                try:
                    media_tensor = self._preprocess_tensor(media_tensor)
                except (RuntimeError, TypeError) as e:
                    raise ValueError(f"Error preprocessing image: {e}")

            media_tensor = media_tensor.unsqueeze(0).to(self.device)  # [C, H, W] -> [1, C, H, W]
            
            self.model.eval()
            with torch.no_grad():
                model_output = self.model(media_tensor)
                
                # Handle different model output formats
                if isinstance(model_output, tuple) and len(model_output) >= 3:
                    # Model with uncertainty estimation
                    logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                    # Use the uncertainty-based probabilities
                    return probs.squeeze().cpu().tolist()
                elif isinstance(model_output, tuple) and len(model_output) == 2:
                    # Model without uncertainty estimation
                    logits, features = model_output
                    probs = F.softmax(logits, dim=1)
                    return probs.squeeze().cpu().tolist()
                else:
                    # Direct logits output
                    probs = F.softmax(model_output, dim=1)
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
        self.model.eval()
        with torch.no_grad():
            for frame in media_tensor:  # frame: [C, H, W]
                if frame.shape[0] not in [1, 3]:
                    raise ValueError(f"Expected 1 or 3 channels for video frame, got shape {frame.shape}")
                
                try:
                    # Try to convert to numpy for albumentations
                    if frame.requires_grad:
                        frame = frame.detach()
                    
                    frame_np = frame.cpu().permute(1, 2, 0).numpy()
                    frame = self._preprocess_numpy(frame_np)
                except Exception:
                    # Fallback to tensor transforms
                    try:
                        frame = self._preprocess_tensor(frame)
                    except (RuntimeError, TypeError) as e:
                        raise ValueError(f"Error preprocessing video frame: {e}")
                
                frame = frame.unsqueeze(0).to(self.device)  # [C, H, W] -> [1, C, H, W]
                
                model_output = self.model(frame)
                
                # Handle different model output formats
                if isinstance(model_output, tuple) and len(model_output) >= 3:
                    # Model with uncertainty estimation
                    logits, features, (probs, epistemic_unc, aleatoric_unc, alpha) = model_output
                    frame_probs.append(probs.squeeze(0))
                elif isinstance(model_output, tuple) and len(model_output) == 2:
                    # Model without uncertainty estimation
                    logits, features = model_output
                    probs = F.softmax(logits, dim=1)
                    frame_probs.append(probs.squeeze(0))
                else:
                    # Direct logits output
                    probs = F.softmax(model_output, dim=1)
                    frame_probs.append(probs.squeeze(0))
        
        avg_probs = torch.stack(frame_probs).mean(dim=0)
        return avg_probs.cpu().tolist()