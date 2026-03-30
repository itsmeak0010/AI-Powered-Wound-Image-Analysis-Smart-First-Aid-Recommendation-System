import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 224
BATCH_SIZE = 16

# Data Augmentation
print("Loading training data...")
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print(f"\nNumber of training samples: {train_data.samples}")
print(f"Number of test samples: {test_data.samples}")
print(f"Number of classes: {train_data.num_classes}")
print(f"Classes: {train_data.class_indices}")

# Load Pretrained EfficientNetV2
print("\nBuilding EfficientNetV2S model...")
base_model = EfficientNetV2S(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Add Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze first layers, fine-tune last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# Train Model with 10 epochs
print("\nTraining EfficientNetV2S model with 10 epochs...")
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data,
    verbose=1
)

# Save model
model.save("efficientnet_model.h5")
print("\nModel saved as efficientnet_model.h5")

# Evaluate on test data
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions
print("\nGenerating predictions on test data...")
test_data.reset()
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# Get class names
class_names = list(test_data.class_indices.keys())

# Print classification report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Plot 1: Training and Validation Accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('efficientnet_training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history graph saved as 'efficientnet_training_history.png'")

# Plot 3: Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Confusion Matrix - EfficientNetV2S', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'efficientnet_confusion_matrix.png'")

# Plot 4: Sample predictions with images
print("\nGenerating sample predictions visualization...")
test_data.reset()
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
fig.suptitle('Sample Predictions from Test Data', fontsize=16, fontweight='bold')

batch = next(test_data)
images = batch[0]
true_labels = np.argmax(batch[1], axis=1)

preds = model.predict(images)
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
print("Sample predictions visualization saved as 'efficientnet_sample_predictions.png'")

print("\n✓ Training completed successfully!")
