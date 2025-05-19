#!/bin/bash

# Exit on error
set -e

# Update and install dependencies
sudo apt update && sudo apt install -y git docker.io docker-compose unzip curl
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Clone the repository (replace with your repo URL)
cd ~
git clone https://github.com/YOUR_USERNAME/pincheck.git
cd pincheck

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip
sudo ./aws/install

# Build and start the containers
docker-compose -f docker-compose.prod.yml up --build -d

echo "✅ EC2 initialization complete. App is deploying via Docker."