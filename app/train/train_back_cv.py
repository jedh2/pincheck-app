# train_back_cv.py with ResNet34 and 384x384 resolution
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "dataset/back"
num_classes = 2
num_epochs = 20
batch_size = 16
num_folds = 5
early_stopping_patience = 3

transform = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)
targets = [label for _, label in dataset]

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_accuracies = []
best_acc = 0.0

os.makedirs("results/confusion_matrices", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
    print(f"Fold {fold + 1}/{num_folds}")
    fold_start = time.time()

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

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

    train_losses = []
    val_losses = []
    val_accuracies = []

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
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

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

        avg_val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    metrics_text = f"Precision: {precision:.2f}  |  Recall: {recall:.2f}  |  F1 Score: {f1:.2f}"
    plt.title(f"Confusion Matrix - Fold {fold + 1}\n{metrics_text}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrices/back_fold{fold + 1}_confusion_matrix.png")
    plt.close()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"model_back_best.pth")

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

    plt.title(f"Training Progress - Fold {fold + 1}")
    plt.tight_layout()
    plt.savefig(f"results/metrics/back_fold{fold + 1}_training_plot.png")
    plt.close()

    fold_accuracies.append(val_acc)
    fold_end = time.time()
    print(f"🕒 Fold {fold + 1} Duration: {(fold_end - fold_start):.2f} seconds")

end_time = time.time()
print(f"✅ Total Training Time: {(end_time - start_time)/60:.2f} minutes")
print(f"📊 Average Accuracy across {num_folds} folds: {np.mean(fold_accuracies):.4f}")
