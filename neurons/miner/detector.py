import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ConvNeXtViTAttention(nn.Module):
    """Model architecture from the training script"""
    def __init__(self):
        super(ConvNeXtViTAttention, self).__init__()
        # Load pre-trained ConvNeXt
        self.convnext = torch.hub.load('pytorch/vision', 'convnext_base', weights='IMAGENET1K_V1')
        self.convnext.classifier[2] = nn.Identity()  # Remove classification head
        self.convnext_features = 1024  # ConvNeXt-base output dimension

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_features = 768  # ViT-base output dimension

        # Project ViT features to match ConvNeXt dimension
        self.vit_projection = nn.Linear(self.vit_features, self.convnext_features)

        # Attention module to fuse ConvNeXt and ViT features
        self.attention = nn.MultiheadAttention(embed_dim=self.convnext_features, num_heads=8)
        self.fc1 = nn.Linear(self.convnext_features + self.convnext_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Binary classification output (raw logits)

    def forward(self, x):
        # ConvNeXt forward
        convnext_out = self.convnext(x)  # Shape: (batch, 1024)
        convnext_out = convnext_out.unsqueeze(0)  # Shape: (1, batch, 1024) for attention

        # ViT forward
        vit_out = self.vit(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token, shape: (batch, 768)
        vit_out = self.vit_projection(vit_out)  # Project to 1024, shape: (batch, 1024)
        vit_out = vit_out.unsqueeze(0)  # Shape: (1, batch, 1024)

        # Attention: use ConvNeXt features as query/key, ViT as value
        attn_output, _ = self.attention(convnext_out, convnext_out, vit_out)  # Shape: (1, batch, 1024)
        attn_output = attn_output.squeeze(0)  # Shape: (batch, 1024)

        # Concatenate ConvNeXt and attention-enhanced features
        combined = torch.cat((convnext_out.squeeze(0), attn_output), dim=1)  # Shape: (batch, 1024+1024)

        # Fully connected layers
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # No sigmoid - BCEWithLogitsLoss applies it internally
        return x

class Detector:
    def __init__(self, model_path: str = None, device: str | None = None):
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Look for the best model from training
        if model_path is None:
            # Check for different possible model paths from the training script
            possible_paths = [
                "best_binary_classifier.pth",  # Best model saved during training
                "final_binary_classifier.pth",  # Final model saved after training
                "binary_checkpoints/checkpoint_best.pth",  # Best checkpoint
                "binary_checkpoints/checkpoint_latest.pth",  # Latest checkpoint
            ]
            
            self.model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_path = path
                    break
            
            if self.model_path is None:
                raise FileNotFoundError(f"No model file found. Tried: {possible_paths}")
        else:
            self.model_path = model_path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Initialize model
        self.model = ConvNeXtViTAttention().to(self.device)
        
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
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_val_mcc' in checkpoint:
                    print(f"Best validation MCC: {checkpoint['best_val_mcc']:.4f}")
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from distributed training)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            print("Removed 'module.' prefix from checkpoint keys")
        
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded model from {self.model_path}")
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
                raise
        
        self.model.eval()
        
        # Define preprocessing pipeline - same as training script
        self.image_size = 224
        
        # Define preprocessing transforms to match training
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Fallback torchvision transforms for tensor input (matches training transforms)
        self.tensor_transforms = transforms.Compose([
            transforms.Lambda(lambda x: x.float() / 255.0 if x.max() > 1.0 else x.float()),  # FloatNormalize equivalent
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Class names for binary classification
        self.class_names = ["synthetic", "real"]  # 0: synthetic, 1: real
    
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
        """
        Detect if media is real or synthetic.
        
        Args:
            media_tensor: Input tensor
            modality: "image" or "video"
            
        Returns:
            List of probabilities [prob_synthetic, prob_real] for binary classification
        """
        media_tensor = media_tensor.to(self.device)
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
                # Get raw logits from model (binary classification)
                logits = self.model(media_tensor)  # Shape: (1,) or (1, 1)
                
                # Handle different output shapes
                if logits.dim() > 1:
                    logits = logits.squeeze()
                
                # Convert logits to probabilities using sigmoid (for binary classification)
                prob_real = torch.sigmoid(logits).item()  # Probability of being real
                prob_synthetic = 1.0 - prob_real  # Probability of being synthetic
                
                return [prob_synthetic, prob_real]

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
                
                # Get raw logits from model (binary classification)
                logits = self.model(frame)  # Shape: (1,) or (1, 1)
                
                # Handle different output shapes
                if logits.dim() > 1:
                    logits = logits.squeeze()
                
                # Convert logits to probabilities using sigmoid
                prob_real = torch.sigmoid(logits).item()
                prob_synthetic = 1.0 - prob_real
                
                frame_probs.append([prob_synthetic, prob_real])
        
        # Average probabilities across all frames
        avg_probs = torch.tensor(frame_probs).mean(dim=0)
        return avg_probs.tolist()

    def predict_class(self, media_tensor: torch.Tensor, modality: str = "image") -> str:
        """
        Predict the class of the media.
        
        Args:
            media_tensor: Input tensor
            modality: "image" or "video"
            
        Returns:
            Class name: "real" or "synthetic"
        """
        probs = self.detect(media_tensor, modality)
        predicted_class_idx = probs.index(max(probs))
        return self.class_names[predicted_class_idx]
    
    def get_confidence(self, media_tensor: torch.Tensor, modality: str = "image") -> float:
        """
        Get the confidence of the prediction.
        
        Args:
            media_tensor: Input tensor
            modality: "image" or "video"
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        probs = self.detect(media_tensor, modality)
        return max(probs)