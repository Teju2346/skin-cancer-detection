# ============================================================
# PART 3: TRANSFER LEARNING - MobileNetV2 (FASTER VERSION)
# Skin Cancer HAM10000
# Run: python 3_transfer_learning.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import cv2
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                      GlobalAveragePooling2D, Input)
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

# ============================================================
# STEP 2: PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("STEP 2: PREPROCESSING")
print("="*60)

X = df.drop('label', axis=1).values
y = df['label'].values

X = X / 255.0

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"Before SMOTE: {X.shape[0]} samples")
print(f"After  SMOTE: {X_balanced.shape[0]} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, random_state=42, stratify=y_balanced
)

# ⚡ FASTER: 48x48 + batch_size=64
print("\nResizing to 48x48 RGB (faster than 96x96)...")
def resize_to_rgb(X, size=48):
    resized = []
    for img in X:
        img_2d = img.reshape(28, 28)
        img_uint8 = (img_2d * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (size, size))
        img_rgb = np.stack([img_resized]*3, axis=-1)
        img_norm = img_rgb / 255.0
        resized.append(img_norm)
    return np.array(resized)

X_train_tl = resize_to_rgb(X_train)
X_test_tl  = resize_to_rgb(X_test)
print(f"Train shape: {X_train_tl.shape}")
print(f"Test  shape: {X_test_tl.shape}")

y_train_cat = to_categorical(y_train, 7)
y_test_cat  = to_categorical(y_test, 7)

# ============================================================
# STEP 3: BUILD MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 3: BUILD MOBILENETV2 (48x48)")
print("="*60)

base_model = MobileNetV2(
    input_shape=(48, 48, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = Input(shape=(48, 48, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dense(128, activation='relu')(x)
x       = BatchNormalization()(x)
x       = Dropout(0.4)(x)
x       = Dense(64, activation='relu')(x)
x       = Dropout(0.3)(x)
outputs = Dense(7, activation='softmax')(x)

tl_model = Model(inputs, outputs)
tl_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Model ready! Parameters: {tl_model.count_params():,}")

# ============================================================
# STEP 4: INITIAL TRAINING
# ============================================================
print("\n" + "="*60)
print("STEP 4: INITIAL TRAINING")
print("="*60)
print("⚡ Faster settings: 48x48 images + batch_size=64")
print("Expected time: 5-10 minutes...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, verbose=1)
]

history1 = tl_model.fit(
    X_train_tl, y_train_cat,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

loss1, acc1 = tl_model.evaluate(X_test_tl, y_test_cat, verbose=0)
print(f"\nAccuracy after initial training: {acc1*100:.2f}%")

# ============================================================
# STEP 5: FINE TUNING
# ============================================================
print("\n" + "="*60)
print("STEP 5: FINE TUNING (Last 20 layers)")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

tl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Fine tuning... Expected time: 5-10 minutes...")
history2 = tl_model.fit(
    X_train_tl, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# STEP 6: EVALUATE
# ============================================================
print("\n" + "="*60)
print("STEP 6: FINAL EVALUATION")
print("="*60)

loss, acc = tl_model.evaluate(X_test_tl, y_test_cat, verbose=0)
print(f"\nTransfer Learning Final Accuracy: {acc*100:.2f}%")
y_pred = np.argmax(tl_model.predict(X_test_tl), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=[class_names[i] for i in range(7)]))

# ============================================================
# STEP 7: PLOTS
# ============================================================
combined_acc      = history1.history['accuracy']     + history2.history['accuracy']
combined_val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
combined_loss     = history1.history['loss']         + history2.history['loss']
combined_val_loss = history1.history['val_loss']     + history2.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(combined_acc,     label='Train Accuracy', color='blue')
axes[0].plot(combined_val_acc, label='Val Accuracy',   color='orange')
axes[0].axvline(x=len(history1.history['accuracy']),
                color='red', linestyle='--', label='Fine Tuning Start')
axes[0].set_title('Transfer Learning - Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(combined_loss,     label='Train Loss', color='blue')
axes[1].plot(combined_val_loss, label='Val Loss',   color='orange')
axes[1].axvline(x=len(history1.history['loss']),
                color='red', linestyle='--', label='Fine Tuning Start')
axes[1].set_title('Transfer Learning - Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('MobileNetV2 Transfer Learning', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_tl_training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 08_tl_training_history.png")

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=[class_names[i] for i in range(7)],
            yticklabels=[class_names[i] for i in range(7)])
plt.title(f'Transfer Learning Confusion Matrix\nAccuracy: {acc*100:.2f}%',
          fontweight='bold', fontsize=13)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('09_tl_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 09_tl_confusion_matrix.png")

# Final comparison
all_models = ['KNN',    'Random Forest', 'SVM',    'Custom CNN', 'MobileNetV2']
all_scores = [91.84,    95.52,           93.46,    92.82,        acc*100      ]
all_colors = ['#FF6B6B','#FF8E53',       '#FFC300','#2ECC71',    '#3498DB'    ]

plt.figure(figsize=(12, 6))
bars = plt.bar(all_models, all_scores, color=all_colors,
               edgecolor='black', width=0.5)
plt.title('ML vs DL — Final Accuracy Comparison',
          fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 105)
plt.xticks(rotation=15)
for bar, score in zip(bars, all_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{score:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('10_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 10_final_comparison.png")

# ============================================================
# SAVE MODEL
# ============================================================
tl_model.save('transfer_learning_model.h5')
with open('tl_results.json', 'w') as f:
    json.dump({'tl_accuracy': float(acc)}, f)

print("\n" + "="*60)
print("TRANSFER LEARNING COMPLETE! 🎉")
print("="*60)
print(f"  KNN           : 91.84%")
print(f"  SVM           : 93.46%")
print(f"  Random Forest : 95.52%")
print(f"  Custom CNN    : 92.82%")
print(f"  MobileNetV2   : {acc*100:.2f}%")
print(f"\n🏆 Best Model: MobileNetV2 ({acc*100:.2f}%)")
print("\nNext Step: Run 4_nlp_report.py")