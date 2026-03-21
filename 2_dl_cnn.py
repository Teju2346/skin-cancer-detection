# ============================================================
# PART 2: DEEP LEARNING - CUSTOM CNN
# Skin Cancer HAM10000 - Convolutional Neural Network
# Run: python 2_dl_cnn.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("="*60)
print("STEP 1: LOADING DATASET")
print("="*60)

df = pd.read_csv('hmnist_28_28_L.csv')
print(f"Dataset shape: {df.shape}")

class_names = {
    0: 'akiec', 1: 'bcc', 2: 'bkl',
    3: 'df',    4: 'mel', 5: 'nv', 6: 'vasc'
}
class_fullnames = {
    0: 'Actinic Keratosis',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular Lesion'
}

# ============================================================
# STEP 2: VISUALIZE SAMPLE IMAGES
# ============================================================
print("\n" + "="*60)
print("STEP 2: VISUALIZE SAMPLE IMAGES")
print("="*60)

fig, axes = plt.subplots(2, 7, figsize=(16, 5))
for label in range(7):
    samples = df[df['label'] == label].head(2)
    for i, (_, row) in enumerate(samples.iterrows()):
        pixels = row.drop('label').values.reshape(28, 28)
        axes[i][label].imshow(pixels, cmap='gray')
        axes[i][label].set_title(f"{class_names[label]}\n{class_fullnames[label][:10]}..",
                                  fontsize=7)
        axes[i][label].axis('off')
plt.suptitle('Sample Images - All 7 Skin Lesion Classes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_sample_images.png")

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("STEP 3: PREPROCESSING")
print("="*60)

X = df.drop('label', axis=1).values
y = df['label'].values

# Normalize
X = X / 255.0

# Apply SMOTE before reshaping
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"Before SMOTE: {X.shape[0]} samples")
print(f"After  SMOTE: {X_balanced.shape[0]} samples")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, random_state=42, stratify=y_balanced
)

# Reshape for CNN (28x28x1)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test.reshape(-1, 28, 28, 1)

# One hot encode labels
y_train_cat = to_categorical(y_train, 7)
y_test_cat  = to_categorical(y_test, 7)

print(f"\nTrain shape: {X_train_cnn.shape}")
print(f"Test  shape: {X_test_cnn.shape}")

# ============================================================
# STEP 4: BUILD CNN MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 4: BUILD CNN MODEL")
print("="*60)

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),

    # Fully connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# STEP 5: TRAIN CNN MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 5: TRAIN CNN MODEL")
print("="*60)
print("Training... (this may take 10-20 minutes)")

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
]

history = model.fit(
    X_train_cnn, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# STEP 6: EVALUATE CNN MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 6: EVALUATE CNN MODEL")
print("="*60)

loss, acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"\nCNN Test Accuracy: {acc*100:.2f}%")
print(f"CNN Test Loss    : {loss:.4f}")

y_pred = np.argmax(model.predict(X_test_cnn), axis=1)

print("\nCNN Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[class_names[i] for i in range(7)]))

# ============================================================
# STEP 7: PLOT TRAINING HISTORY
# ============================================================
print("\n" + "="*60)
print("STEP 7: TRAINING HISTORY")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
axes[0].set_title('CNN - Accuracy over Epochs', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
axes[1].plot(history.history['val_loss'], label='Val Loss', color='orange')
axes[1].set_title('CNN - Loss over Epochs', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('CNN Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('06_cnn_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_cnn_training_history.png")

# ============================================================
# STEP 8: CONFUSION MATRIX
# ============================================================
print("\n" + "="*60)
print("STEP 8: CNN CONFUSION MATRIX")
print("="*60)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=[class_names[i] for i in range(7)],
            yticklabels=[class_names[i] for i in range(7)])
plt.title(f'CNN Confusion Matrix\nAccuracy: {acc*100:.2f}%',
          fontweight='bold', fontsize=13)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('07_cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_cnn_confusion_matrix.png")

# ============================================================
# STEP 9: SAVE MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 9: SAVE MODEL")
print("="*60)

model.save('cnn_model.h5')
print("Saved: cnn_model.h5")

# Save accuracy for later comparison
import json
results = {'cnn_accuracy': float(acc)}
with open('cnn_results.json', 'w') as f:
    json.dump(results, f)
print("Saved: cnn_results.json")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("CNN TRAINING COMPLETE! 🎉")
print("="*60)
print(f"CNN Test Accuracy: {acc*100:.2f}%")
print("\nSaved Files:")
print("  cnn_model.h5")
print("  cnn_results.json")
print("  05_sample_images.png")
print("  06_cnn_training_history.png")
print("  07_cnn_confusion_matrix.png")
print("\nNext Step: Run 3_transfer_learning.py")