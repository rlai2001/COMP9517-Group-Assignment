import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")
epochs = df["epoch"]

plt.figure()
plt.plot(epochs, df["train_acc"], label="Train Accuracy")
plt.plot(epochs, df["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_epoch.png")

plt.figure()
plt.plot(epochs, df["train_loss"], label="Train Loss")
plt.plot(epochs, df["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_epoch.png")

print("The images have been saved as accuracy_vs_epochs.png and loss_vs_epochs.png")
