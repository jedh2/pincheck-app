# PinCheck: Disney Pin Classifier

A web-based tool that uses machine learning to classify Disney pins as real or fake using front and back images. Includes Grad-CAM visualization, S3 model storage, HTTPS, and CI/CD with GitHub Actions.

## Features
- Upload front and back images
- Classify pins as real or fake
- Grad-CAM overlay explanations
- S3-hosted model loading via boto3
- CI/CD deploy to EC2
- HTTPS via Nginx + Certbot

## How to Deploy
1. Fork or clone this repo
2. Push to `main` to trigger CI/CD
3. EC2 instance will auto-pull repo and run with Docker

## Environment Setup (on EC2)
Run:
```bash
bash init_ec2.sh
```

## S3 Setup
- Upload `model_front.pth` and `model_back.pth` to S3
- Set credentials in EC2 or GitHub secrets

## GitHub Secrets Required
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

---

Contributions welcome!