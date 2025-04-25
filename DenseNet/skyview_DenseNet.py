import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import pandas as pd

# 开始进行数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 对测试集集进行基础预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, avg_loss, f1, precision, recall, all_labels, all_preds

def main():
    data_dir = './Aerial_Landscapes'
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                pbar.set_postfix(loss=running_loss / len(train_loader), acc=correct / total)

        scheduler.step()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        test_acc, test_loss, test_f1, test_precision, test_recall, y_true, y_pred = evaluate(model, test_loader, device, criterion)

        print(f"\n Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Acc: {train_acc:.4f}, Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
        print(f"Test  - Acc: {test_acc:.4f}, Loss: {test_loss:.4f}, F1: {test_f1:.4f}")

        history.append({
            'epoch': epoch+1,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall
        })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'temp_best_model.pth')
            torch.save(torch.tensor(y_true), 'y_true.pt')
            torch.save(torch.tensor(y_pred), 'y_pred.pt')

    os.rename('temp_best_model.pth', 'best_model.pth')
    pd.DataFrame(history).to_csv("metrics.csv", index=False)

    print(f"The best test F1 score: {best_f1:.4f}, Epoch {best_epoch}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()