import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = './Aerial_Landscapes'
model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.densenet121(weights=None)  # 加载模型
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
class_names = dataset.classes

def add_noise(img):
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    noise = np.random.normal(0, 0.1, img_np.shape)
    noisy = np.clip(img_np + noise, 0, 1)
    noisy_tensor = torch.tensor(noisy).float().permute(2, 0, 1)
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])(noisy_tensor)

def add_blur(img):
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    blur = cv2.GaussianBlur((img_np * 255).astype(np.uint8), (9, 9), 5)
    blur = blur.astype(np.float32) / 255.0
    blur_tensor = torch.tensor(blur).float().permute(2, 0, 1)
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])(blur_tensor)

def add_occlusion(img):
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    h, w, _ = img_np.shape
    x0, y0 = w // 4, h // 4
    img_np[y0:y0+50, x0:x0+50, :] = 0
    occ_tensor = torch.tensor(img_np).float().permute(2, 0, 1)
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])(occ_tensor)

# 统计准确率
correct_total = 0
correct_noise = 0
correct_blur = 0
correct_occ = 0


with torch.no_grad():
    for img, label in tqdm(dataset, desc="Processing Images", ncols=80):
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

        if pred == label:
            correct_total += 1

            # Re-predict the perturbed image
            noisy = add_noise(img).unsqueeze(0).to(device)
            blur = add_blur(img).unsqueeze(0).to(device)
            occ  = add_occlusion(img).unsqueeze(0).to(device)

            if model(noisy).argmax(dim=1).item() == label:
                correct_noise += 1
            if model(blur).argmax(dim=1).item() == label:
                correct_blur += 1
            if model(occ).argmax(dim=1).item() == label:
                correct_occ += 1

total = correct_total
acc_original = 100.0
acc_noise = 100.0 * correct_noise / total
acc_blur = 100.0 * correct_blur / total
acc_occ  = 100.0 * correct_occ / total

print(f"\nThe number of correct images predicted in the original way: {total}")
print(f"Original accuracy rate: {acc_original:.1f}%")
print(f"Noise accuracy rate: {acc_noise:.1f}%")
print(f"Blur accuracy rate: {acc_blur:.1f}%")
print(f"Occlusion accuracy rate: {acc_occ:.1f}%")

# 进行绘图
accs = [acc_original, acc_noise, acc_blur, acc_occ]
labels = ['Original', 'Noise', 'Blur', 'Occlusion']
colors = ['green', 'red', 'dodgerblue', 'orange']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, accs, color=colors)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}%', ha='center', fontsize=10)

plt.title("Accuracy under Different Perturbations")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("perturbation_accuracy.png")
plt.show()
