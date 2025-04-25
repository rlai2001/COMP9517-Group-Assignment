import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import random
from collections import defaultdict
from torchcam.methods import GradCAM
import pandas as pd

data_dir = './Aerial_Landscapes'
model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载最佳模型
model = models.densenet121(weights=None)
num_classes = len(os.listdir(data_dir))
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(data_dir, transform=transform)
class_names = dataset.classes

# 找出原始预测正确的图像
print("Filtering the correctly predicted images...")
correct_indices = []
with torch.no_grad():
    for idx in range(len(dataset)):
        image, true_label = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        if pred_label == true_label:
            correct_indices.append(idx)

print(f"{len(correct_indices)} images were originally predicted correctly.")

# 从中选取每类各 1 张，共 15 张图像
label_to_correct = defaultdict(list)
for idx in correct_indices:
    _, label = dataset[idx]
    label_to_correct[label].append(idx)

selected_indices = []
labels_covered = set()
all_labels = sorted(label_to_correct.keys())
random.shuffle(all_labels)
for label in all_labels:
    if len(selected_indices) >= 15:
        break
    if label_to_correct[label]:
        idx = random.choice(label_to_correct[label])
        selected_indices.append(idx)
        labels_covered.add(label)

print(f"Finally selected {len(selected_indices)}  images from the originally correctly predicted ones, covering {len(labels_covered)}")

selected_subset = torch.utils.data.Subset(dataset, selected_indices)
test_loader = DataLoader(selected_subset, batch_size=1, shuffle=False)

# 进行GradCAM
cam_extractor = GradCAM(model, target_layer="features.denseblock4")

def add_noise(img):
    noise = np.random.normal(0, 0.1, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_blur(img):
    img_blur = cv2.GaussianBlur(img, (9, 9), 5)
    return img_blur

def add_occlusion(img):
    img_occ = img.copy()
    h, w, _ = img.shape
    x0, y0 = w // 4, h // 4
    img_occ[y0:y0+50, x0:x0+50, :] = 0
    return img_occ

def apply_colored_cam(image_np, activation_map):
    activation_map = F.interpolate(activation_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze()
    activation_map = activation_map - activation_map.min()
    activation_map = activation_map / (activation_map.max() + 1e-6)
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map.numpy()), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(image_np * 255), 0.5, heatmap, 0.5, 0)
    return overlay

# 图像保存
def save_grid_comparison(imgs, labels, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=10, pad=10)
        axes[i].axis('off')
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 执行 Grad-CAM 并生成拼图和记录预测
output_root = "gradcam_compare"
os.makedirs(f"{output_root}/grids", exist_ok=True)
prediction_records = []

print("Starting to process perturbations and Grad-CAM, and create the collage...")

for idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    img_tensor = inputs.squeeze().cpu()
    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

    grid_images = []
    grid_labels = []
    pred_row = {"img": f"img_{idx}_grid.png"}

    def get_cam_overlay(image_np, tag):
        image_tensor = transform(Image.fromarray((image_np * 255).astype(np.uint8))).unsqueeze(0).to(device)
        scores = model(image_tensor)
        class_id = scores.argmax().item()
        activation_map = cam_extractor(class_id, scores)[0].cpu()
        overlay = apply_colored_cam(image_np, activation_map)
        pred_row[tag] = class_names[class_id]
        return overlay

    # 原图
    grid_images.append(get_cam_overlay(img, "original"))
    grid_labels.append(f"Original\nPred: {pred_row['original']}")

    # 高斯噪声
    noisy_img = add_noise(img)
    grid_images.append(get_cam_overlay(noisy_img, "noise"))
    grid_labels.append(f"Noise\nPred: {pred_row['noise']}")

    # 模糊
    blur_img = add_blur((img * 255).astype(np.uint8))
    grid_images.append(get_cam_overlay(blur_img.astype(np.uint8)/255.0, "blur"))
    grid_labels.append(f"Blur\nPred: {pred_row['blur']}")

    # 遮挡
    occ_img = add_occlusion((img * 255).astype(np.uint8))
    grid_images.append(get_cam_overlay(occ_img.astype(np.uint8)/255.0, "occlusion"))
    grid_labels.append(f"Occlusion\nPred: {pred_row['occlusion']}")

    save_grid_comparison(grid_images, grid_labels, f"{output_root}/grids/img_{idx}_grid.png")
    prediction_records.append(pred_row)

pred_df = pd.DataFrame(prediction_records)
pred_df.to_csv(f"{output_root}/grid_predictions.csv", index=False)

# 最常见的错误类对
pred_df["noise_changed"] = pred_df["original"] != pred_df["noise"]
pred_df["blur_changed"] = pred_df["original"] != pred_df["blur"]
pred_df["occlusion_changed"] = pred_df["original"] != pred_df["occlusion"]

error_pairs = pd.concat([
    pred_df[pred_df["noise_changed"]][["original", "noise"]].rename(columns={"noise": "pred"}),
    pred_df[pred_df["blur_changed"]][["original", "blur"]].rename(columns={"blur": "pred"}),
    pred_df[pred_df["occlusion_changed"]][["original", "occlusion"]].rename(columns={"occlusion": "pred"}),
])

error_counts = error_pairs.value_counts().reset_index(name="count").sort_values("count", ascending=False)
error_counts.head(15).to_csv(f"{output_root}/top15_confused_pairs.csv", index=False)

print("The most common misclassified class pairs have been saved to top15_confused_pairs.csv.")