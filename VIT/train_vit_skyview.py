import os
import csv
import torch
import numpy as np
from torch import nn
from transformers import ViTForImageClassification
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from prepare_skyview_data import get_skyview_dataloaders

# ==== 设置设备 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== 参数设置 ====
data_dir = "skyview_dataset"  # 指向解压后的数据路径
EPOCHS = 15
PATIENCE = 3
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "best_vit_model.pt"
RESULT_CSV = "epoch_metrics.csv"
batch_size = 32
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==== 加载数据 ====
train_loader, val_loader, class_names = get_skyview_dataloaders(data_dir, batch_size)

# ==== 构建模型 ====
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names),
    id2label={str(i): name for i, name in enumerate(class_names)},
    label2id={name: str(i) for i, name in enumerate(class_names)}
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
wait = 0
best_epoch = 0
train_losses, val_losses = [], []
train_accuracies, train_f1s = [], []
val_accuracies, val_f1s = [], []

with open(RESULT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc", "Train_F1", "Val_F1"])

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        # ==== 训练 ====
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for batch in tqdm(train_loader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(pred)
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)

        # ==== 验证 ====
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)

        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, train_acc, val_acc, train_f1, val_f1])

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        # ==== Checkpoint 保存 ====
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping triggered!")
                break

# ==== 载入最佳模型 ====
model.load_state_dict(torch.load(BEST_MODEL_PATH))
print(f"Loaded best model from epoch {best_epoch+1}")

# ==== 可视化训练过程 ====
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics_full.png")
plt.show()

# ==== 混淆矩阵 ====
y_true, y_pred = np.array(all_labels), np.array(all_preds)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("Training complete. Results saved to training_metrics_full.png and confusion_matrix.png and epoch_metrics.csv")
