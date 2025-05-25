import torch

print("Starting model check...")  # Debug print

model_path = 'xray_classifier.pth'

try:
    model_weights = torch.load(model_path, map_location='cpu')
    print("Model loaded successfully!")
    print("Model keys:", model_weights.keys())  # Lists the layers/parameters saved
except Exception as e:
    print(f"Error loading model: {e}")



