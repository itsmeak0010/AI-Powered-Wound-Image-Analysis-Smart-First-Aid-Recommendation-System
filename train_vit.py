import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10  # 10 epochs as requested

# GPU/Device settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
CHECKPOINT_DIR = "vit_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Loading dataset...")
# Transforms with normalization for ViT
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
test_dataset = datasets.ImageFolder("dataset/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_classes = len(train_dataset.classes)
class_names = train_dataset.classes

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")

# Load pretrained ViT
print("\nLoading Vision Transformer model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 Device Configuration:")
print(f"   Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Track metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_checkpoint_path = None

# Training loop
print(f"\n📊 Training ViT for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_acc = correct / total_samples
    avg_train_loss = total_loss / len(train_loader)
    train_accuracies.append(train_acc)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    val_acc = correct / total_samples
    avg_val_loss = val_loss / len(test_loader)
    val_accuracies.append(val_acc)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
    
    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"vit_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_checkpoint_path = checkpoint_path
        torch.save(model.state_dict(), "vit_model.pth")
        print(f"✓ Best model updated (Val Acc: {val_acc:.4f})")

print(f"\n✓ Training completed! Best validation accuracy: {best_val_acc:.4f}")

# Final Evaluation
print("\n" + "="*60)
print("📊 FINAL EVALUATION ON TEST DATASET")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        predictions = outputs.argmax(1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        all_preds.extend(predictions)
        all_labels.extend(labels_np)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
test_accuracy = np.mean(all_preds == all_labels)

print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({int(test_accuracy*len(all_labels))}/{len(all_labels)} samples)")
print(f"✓ Model saved: vit_model.pth")
print(f"✓ Checkpoints saved in: {CHECKPOINT_DIR}/")
print(f"✓ Best validation accuracy: {best_val_acc:.4f}")

# Classification Report
print("\n📈 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Plot 1: Training and Validation Accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Training Accuracy', linewidth=2, marker='o', markersize=6)
plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation Accuracy', linewidth=2, marker='s', markersize=6)
plt.title('ViT Model Accuracy over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_losses, label='Training Loss', linewidth=2, marker='o', markersize=6)
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=6)
plt.title('ViT Model Loss over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vit_training_history.png', dpi=300, bbox_inches='tight')
print("\n✓ Training history graph saved as 'vit_training_history.png'")

# Plot 3: Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Confusion Matrix - Vision Transformer', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('vit_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'vit_confusion_matrix.png'")

# Plot 4: Sample predictions with images
print("\nGenerating sample predictions visualization...")
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Sample Predictions from Test Data - Vision Transformer', fontsize=16, fontweight='bold')

# Get a batch from test loader
test_iter = iter(test_loader)
images, labels = next(test_iter)
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images).logits
    preds = outputs.argmax(1).cpu().numpy()

images_np = images.cpu().numpy()
labels_np = labels.cpu().numpy()

# Denormalize images
images_np = (images_np * 0.5) + 0.5  # Reverse normalization
images_np = np.clip(images_np, 0, 1)
images_np = np.transpose(images_np, (0, 2, 3, 1))

for i, ax in enumerate(axes.flat):
    if i < len(images_np):
        ax.imshow(images_np[i])
        true_label = class_names[labels_np[i]]
        pred_label = class_names[preds[i]]
        
        color = 'green' if labels_np[i] == preds[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontweight='bold', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('vit_sample_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Sample predictions visualization saved as 'vit_sample_predictions.png'")

print("\n" + "="*60)
print("✅ ViT TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
