# train_front_masked.py: Train on real Disney pins after background removal via U-Net
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# ---------------- CONFIG ----------------
side = "front"
data_dir = f"masked_dataset/{side}"
results_dir = f"results/{side}_masked"
os.makedirs(f"{results_dir}/confusion_matrices", exist_ok=True)
os.makedirs(f"{results_dir}/metrics", exist_ok=True)
os.makedirs(f"{results_dir}/predictions", exist_ok=True)
os.makedirs(f"{results_dir}/misclassified", exist_ok=True)

num_classes = 2
num_epochs = 20
batch_size = 16
early_stopping_patience = 3
num_trials = 2

# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Masked images should already have white/transparent background removed ---
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
targets = [label for _, label in dataset]

summary_rows = []
best_trial_acc = 0.0

for trial in range(1, num_trials + 1):
    print(f"\n🔁 Trial {trial}/{num_trials}")
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=0.1, stratify=targets, random_state=trial * 42
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for layer in [model.layer2, model.layer3, model.layer4]:
        for param in layer.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, val_accuracies = [], [], []
    all_preds, all_labels = [], []
    misclassified = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds.clear()
        all_labels.clear()
        misclassified.clear()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(preds)):
                    if preds[i] != labels[i]:
                        misclassified.append((inputs[i].cpu(), preds[i].item(), labels[i].item()))

        avg_val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"model_{side}_masked_best_trial{trial}.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break

    for i, (img_tensor, pred, true) in enumerate(misclassified[:12]):
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{results_dir}/misclassified/trial{trial}_mis_{i}_P{pred}_T{true}.jpg")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(f"Trial {trial} | Precision: {precision:.2f} Recall: {recall:.2f} F1: {f1:.2f}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrices/trial{trial}_confusion_matrix.png")
    plt.close()

    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')

    plt.title(f"Training Progress - Trial {trial}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics/trial{trial}_training_plot.png")
    plt.close()

    with open(f"{results_dir}/predictions/trial{trial}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True Label", "Predicted Label"])
        writer.writerows(zip(all_labels, all_preds))

    summary_rows.append([trial, val_acc, precision, recall, f1])

    if val_acc > best_trial_acc:
        best_trial_acc = val_acc
        torch.save(model.state_dict(), f"model_{side}_masked_best_overall.pth")

with open(f"{results_dir}/metrics/summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Trial", "Val Accuracy", "Precision", "Recall", "F1 Score"])
    writer.writerows(summary_rows)

print(f"\n🏁 Best Trial Val Accuracy (masked): {best_trial_acc:.4f}")
