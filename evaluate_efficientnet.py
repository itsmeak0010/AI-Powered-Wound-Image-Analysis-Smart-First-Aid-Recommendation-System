import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("Loading trained EfficientNet model...")
model = load_model("efficientnet_model.h5")

# Load test data
print("Loading test data...")
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

print(f"Number of test samples: {test_data.samples}")
print(f"Classes: {test_data.class_indices}")

# Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(test_data)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
print("\nGenerating predictions...")
test_data.reset()
predictions = model.predict(test_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

class_names = list(test_data.class_indices.keys())

# Classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Confusion Matrix - EfficientNetV2S', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved as 'efficientnet_confusion_matrix.png'")

# Sample predictions
print("\nGenerating sample predictions...")
test_data.reset()
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
fig.suptitle('Sample Predictions from Test Data - EfficientNetV2S', fontsize=16, fontweight='bold')

batch = next(test_data)
images = batch[0]
true_labels = np.argmax(batch[1], axis=1)

preds = model.predict(images, verbose=0)
pred_labels = np.argmax(preds, axis=1)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', color=color, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig('efficientnet_sample_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Sample predictions saved as 'efficientnet_sample_predictions.png'")

print("\n✓ Evaluation complete!")
