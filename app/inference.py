
import os
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from torchvision import transforms
from PIL import Image
from app.gradcam import generate_gradcam
import urllib.request
import traceback

# Public S3 HTTP fallback URLs
S3_URLS = {
    'front': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_front.pth',
    'back': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_back.pth'
}

# Where model will be stored inside the container/app
model_paths = {
    'front': 'app/model/model_front.pth',
    'back': 'app/model/model_back.pth'
}

# S3 bucket/key for authenticated download
s3_bucket = 'pincheck-models'
s3_keys = {
    'front': 'model_front.pth',
    'back': 'model_back.pth'
}

# Control download strategy
USE_S3 = True  # Set to False to force fallback to public HTTP

def download_model_if_needed(side):
    local_path = model_paths[side]
    print(f"🔍 Checking if model for '{side}' exists at: {local_path}")
    if os.path.exists(local_path):
        print(f"✅ Model for {side} already exists.")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"📦 Attempting to download {side} model...")

    try:
        if USE_S3:
            print(f"🔐 Trying S3 download for {side}...")
            s3 = boto3.client('s3')
            s3.download_file(s3_bucket, s3_keys[side], local_path)
            if os.path.exists(local_path):
                print(f"✅ Downloaded {side} model from S3 to {local_path}.")
            else:
                print(f"❌ S3 download claimed success but file does not exist at {local_path}.")
        else:
            raise NoCredentialsError()
    except NoCredentialsError:
        try:
            print(f"🌐 Falling back to public URL for {side}...")
            urllib.request.urlretrieve(S3_URLS[side], local_path)
            if os.path.exists(local_path):
                print(f"✅ Downloaded {side} model from public S3 URL.")
            else:
                print(f"❌ Public download claimed success but file not found at {local_path}.")
        except Exception as e:
            print(f"❌ Public URL fallback failed for {side}: {e}")
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"❌ S3 download failed for {side}: {e}")
        traceback.print_exc()
        raise

def load_model(side):
    print(f"🚀 Loading model for {side}...")
    download_model_if_needed(side)
    print(f"📥 Loading state dict from {model_paths[side]}")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_paths[side], map_location='cpu'))
    model.eval()
    print(f"✅ Model for {side} is ready.")
    return model

def predict_image(file, side):
    print(f"🖼️ Running prediction for {side} image...")
    model = load_model(side)
    image = Image.open(file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()
    label = 'real' if pred == 1 else 'fake'
    print(f"🔎 Prediction for {side}: {label}")

    gradcam_path = generate_gradcam(model, input_tensor, side)
    return label, gradcam_path
