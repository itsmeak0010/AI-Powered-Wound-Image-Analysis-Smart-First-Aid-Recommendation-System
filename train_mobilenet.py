import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
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

print("Loading training data...")
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory("dataset/train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical")
test_data = test_gen.flow_from_directory("dataset/test", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical")

print(f"\nNumber of training samples: {train_data.samples}")
print(f"Number of test samples: {test_data.samples}")
print(f"Number of classes: {train_data.num_classes}")
print(f"Classes: {train_data.class_indices}")

print("\nBuilding MobileNetV3Large model...")
base_model = MobileNetV3Large(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# Train the model with 10 epochs
print("\nTraining MobileNetV3Large model with 10 epochs...")
history = model.fit(train_data, epochs=10, validation_data=test_data, verbose=1)

# Save the model
model.save("mobilenet_model.h5")
print("\nModel saved as mobilenet_model.h5")

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
plt.savefig('mobilenet_training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history graph saved as 'mobilenet_training_history.png'")

# Plot 3: Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Confusion Matrix - MobileNetV3Large', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('mobilenet_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'mobilenet_confusion_matrix.png'")

# Plot 4: Sample predictions with images
print("Generating sample predictions visualization...")
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
plt.savefig('mobilenet_sample_predictions.png', dpi=300, bbox_inches='tight')
print("Sample predictions visualization saved as 'mobilenet_sample_predictions.png'")

print("\n✓ Training completed successfully!")
