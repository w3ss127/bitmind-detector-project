#!/usr/bin/env python3
"""
Simple test script for testing the unlimited deepfake model on specific images
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os
import time

# Model architecture - must match the training script exactly
class EfficientNetDeepfakeDetector(nn.Module):
    """Enhanced EfficientNet-based 3-class deepfake detector - matches training script"""
    
    def __init__(self, num_classes=3, model_name='efficientnet_b4'):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        # Use larger EfficientNet for unlimited data
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Enhanced classifier for unlimited data - EXACT match to training script
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = EfficientNetDeepfakeDetector(num_classes=3, model_name='efficientnet_b4')
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """Preprocess image for the model"""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_image(model, image_path, device):
    """Make prediction on a single image"""
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    print(f"\n🔍 Testing: {image_path}")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Make prediction
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    inference_time = time.time() - start_time
    
    # Get probabilities for each class
    probs = probabilities.cpu().numpy()[0]
    
    print(f"⏱️  Inference time: {inference_time:.4f} seconds")
    print(f"🎯 Prediction: {class_names[predicted_class]}")
    print(f"📊 Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
        confidence = prob * 100
        indicator = "👈" if i == predicted_class else "  "
        print(f"   {class_name}: {confidence:.2f}% {indicator}")
    
    return predicted_class, probs, inference_time

def main():
    print("🚀 Starting Unlimited Deepfake Detector Test")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Model path
    model_path = "best_unlimited_deepfake_detector.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the model file is in the current directory.")
        return
    
    # Load model
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test images
    test_images = ["images/1.jpg", "images/2.jpg"]
    
    print(f"\n📸 Testing {len(test_images)} images...")
    
    total_time = 0
    results = []
    
    for image_path in test_images:
        if os.path.exists(image_path):
            try:
                predicted_class, probs, inference_time = predict_image(model, image_path, device)
                total_time += inference_time
                results.append({
                    'image': image_path,
                    'prediction': predicted_class,
                    'probabilities': probs,
                    'time': inference_time
                })
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
        else:
            print(f"❌ Image not found: {image_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    class_names = ['Real', 'Synthetic', 'Semi-synthetic']
    
    for result in results:
        image_name = os.path.basename(result['image'])
        prediction = class_names[result['prediction']]
        confidence = result['probabilities'][result['prediction']] * 100
        print(f"🖼️  {image_name}: {prediction} ({confidence:.1f}% confidence)")
    
    if results:
        avg_time = total_time / len(results)
        print(f"\n⏱️  Average inference time: {avg_time:.4f} seconds")
        print(f"🎯 Both images processed successfully!")
        
        # Performance check
        if avg_time < 6.0:
            print("✅ Model meets the 6-second timeout requirement!")
        else:
            print("⚠️  Model exceeds 6-second timeout requirement")

if __name__ == "__main__":
    main() 