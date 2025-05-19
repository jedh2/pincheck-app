import os
import torch
import boto3
from botocore.exceptions import NoCredentialsError
from torchvision import transforms
from PIL import Image
from app.gradcam import generate_gradcam
import urllib.request

# Public S3 HTTP fallback URLs
S3_URLS = {
    'front': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_front.pth',
    'back': 'https://pincheck-models.s3.us-east-2.amazonaws.com/model_back.pth'
}

# Where model will be stores inside the container/app
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

def download_model_if_needed(side):
    local_path = model_paths[side]
    if os.path.exists(local_path):
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"📦 Downloading {side} model...")

    try:
        # Try S3 (if credentials are configured)
        s3 = boto3.client('s3')
        s3.download_file(s3_bucket, s3_keys[side], local_path)
        print(f"✅ Downloaded {side} model from S3.")
    except NoCredentialsError:
        print("⚠️ No AWS credentials found. Falling back to public URL.")
        urllib.request.urlretrieve(S3_URLS[side], local_path)
        print(f"✅ Downloaded {side} model from public S3 URL.")
    except Exception as e:
        print(f"❌ Failed to download {side} model: {e}")
        raise


def load_model(side):
    download_model_if_needed(side)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_paths[side], map_location='cpu'))
    model.eval()
    return model


def predict_image(file, side):
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

    gradcam_path = generate_gradcam(model, input_tensor, side)
    return label, gradcam_path