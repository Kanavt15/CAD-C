#!/usr/bin/env python3
"""
Enhanced Lung Cancer Detection Ensemble - Production Deployment
Integrates LUNA16-trained DenseNet with existing ensemble models
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path

class DenseNet169LungCancer(nn.Module):
    """DenseNet-169 for lung cancer detection"""
    def __init__(self, num_classes=2):
        super(DenseNet169LungCancer, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.densenet.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.densenet(x)

class EnhancedLungCancerEnsemble:
    """Production-ready enhanced ensemble system"""
    
    def __init__(self, config_path='enhanced_ensemble_config.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.config = self.load_config(config_path)
        self.transform = self.setup_transforms()
        
    def load_config(self, config_path):
        """Load ensemble configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_primary_model(self):
        """Load the primary LUNA16 DenseNet model"""
        model_path = self.config['model_paths']['luna16_densenet']
        
        if Path(model_path).exists():
            model = DenseNet169LungCancer(num_classes=2).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models['luna16_densenet'] = model
            return True
        return False
    
    def predict(self, image_path):
        """Make prediction on medical image"""
        if 'luna16_densenet' not in self.models:
            if not self.load_primary_model():
                return {'error': 'Primary model not available'}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.models['luna16_densenet'](image_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = output.argmax(dim=1).item()
                confidence = probabilities.max().item()
            
            # Interpret results
            diagnosis = "Normal" if prediction == 0 else "Abnormal - Potential Nodule"
            risk_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            
            return {
                'prediction': prediction,
                'diagnosis': diagnosis,
                'confidence': round(confidence, 4),
                'risk_level': risk_level,
                'model_used': 'luna16_densenet',
                'model_performance': self.config['performance_metrics']['luna16_densenet']
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = EnhancedLungCancerEnsemble()
    
    # Test with sample image
    result = ensemble.predict('test_image.jpg')
    print(json.dumps(result, indent=2))
