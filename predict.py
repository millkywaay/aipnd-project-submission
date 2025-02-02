#!/usr/bin/env python3
import argparse
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn

def get_input_args():
    parser = argparse.ArgumentParser(
        description='Predict the class of an image using a trained ResNet18 model.'
    )
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(checkpoint['class_to_idx'])),  # Output layer size from checkpoint
        nn.LogSoftmax(dim=1)
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """Process an image: resize, crop, normalize, and convert to a tensor."""
    image = Image.open(image_path).convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = preprocess(image)
    return image_tensor

def predict(image_path, model, device, top_k=5):
    """Predict the class of an image using the trained model."""
    model.to(device)
    model.eval()
    
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)
    top_p = top_p.cpu().numpy().squeeze()
    top_class = top_class.cpu().numpy().squeeze()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    if top_k == 1:
        top_labels = [idx_to_class[top_class]]
    else:
        top_labels = [idx_to_class[i] for i in top_class]
    
    return top_p, top_labels

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, device, args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        flower_names = [cat_to_name.get(str(label), str(label)) for label in classes]
    else:
        flower_names = classes
    
    print("\nTop K Predictions:")
    for i in range(len(probs)):
        print(f"{flower_names[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
