import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置参数
data_dir = './Aerial_Landscapes'
model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
print(f"device: {device}")

# 加载模型
model = models.densenet121(weights=None)
num_classes = len(os.listdir(data_dir))
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = ImageFolder(data_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


acc_original = 98.5

# 进行扰动函数
def add_noise_batch(imgs):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = np.clip((imgs * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    noise = np.random.normal(0, 0.1, imgs.shape)
    noisy = np.clip(imgs + noise, 0, 1)
    noisy = (torch.tensor(noisy).permute(0, 3, 1, 2).float() - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return noisy.to(device)

def add_blur_batch(imgs):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = np.clip((imgs * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    blurred = []
    for img in imgs:
        blur = cv2.GaussianBlur((img * 255).astype(np.uint8), (9, 9), 5)
        blur = blur.astype(np.float32) / 255.0
        blurred.append(blur)
    blurred = np.stack(blurred)
    blurred = (torch.tensor(blurred).permute(0, 3, 1, 2).float() - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return blurred.to(device)

def add_occlusion_batch(imgs):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = np.clip((imgs * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    for i in range(len(imgs)):
        h, w, _ = imgs[i].shape
        x0, y0 = w // 4, h // 4
        imgs[i][y0:y0+50, x0:x0+50, :] = 0
    occ = (torch.tensor(imgs).permute(0, 3, 1, 2).float() - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return occ.to(device)

# 对三种扰动进行测试
total = 0
correct_noise = 0
correct_blur = 0
correct_occ = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", ncols=80):
        images, labels = images.to(device), labels.to(device)

        noise_imgs = add_noise_batch(images)
        blur_imgs = add_blur_batch(images)
        occ_imgs = add_occlusion_batch(images)

        pred_noise = model(noise_imgs).argmax(dim=1)
        pred_blur  = model(blur_imgs).argmax(dim=1)
        pred_occ   = model(occ_imgs).argmax(dim=1)

        correct_noise += (pred_noise == labels).sum().item()
        correct_blur  += (pred_blur  == labels).sum().item()
        correct_occ   += (pred_occ   == labels).sum().item()
        total += labels.size(0)

# 输出准确率
acc_noise = 100.0 * correct_noise / total
acc_blur  = 100.0 * correct_blur  / total
acc_occ   = 100.0 * correct_occ   / total

print(f"\nTotal number of test images: {total}")
print(f"Original accuracy rate (provided): {acc_original:.2f}%")
print(f"Noise accuracy rate: {acc_noise:.2f}%")
print(f"Blur accuracy rate: {acc_blur:.2f}%")
print(f"Occlusion accuracy rate: {acc_occ:.2f}%")

# 开始绘制柱形图
accs = [acc_original, acc_noise, acc_blur, acc_occ]
labels = ['Original', 'Noise', 'Blur', 'Occlusion']
colors = ['green', 'red', 'dodgerblue', 'orange']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, accs, color=colors)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1.5, f'{height:.2f}%', ha='center', fontsize=10)

plt.title("Accuracy under Different Perturbations (Best Model)")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("perturbation_accuracy.png")
plt.show()