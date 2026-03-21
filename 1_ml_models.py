# ============================================================
# PART 1: MACHINE LEARNING MODELS
# Skin Cancer HAM10000 - KNN, Random Forest, SVM
# Run: python 1_ml_models.py
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("="*60)
print("STEP 1: LOADING DATASET")
print("="*60)

df = pd.read_csv('hmnist_8_8_L.csv')
print(f"Dataset shape: {df.shape}")
print(f"Classes: {df['label'].value_counts().to_dict()}")

# Class names
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
# STEP 2: EDA - CLASS DISTRIBUTION
# ============================================================
print("\n" + "="*60)
print("STEP 2: EDA - CLASS DISTRIBUTION")
print("="*60)

label_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#DDA0DD','#98D8C8']
bars = plt.bar(
    [class_names[i] for i in range(7)],
    [label_counts.get(i, 0) for i in range(7)],
    color=colors, edgecolor='black'
)
plt.title('Class Distribution - HAM10000 Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Skin Lesion Type')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
for bar, count in zip(bars, [label_counts.get(i, 0) for i in range(7)]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 30,
             str(count), ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_class_distribution.png")

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
print(f"Features shape: {X.shape}")

# Apply SMOTE
print("\nApplying SMOTE to fix class imbalance...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"Before SMOTE: {X.shape[0]} samples")
print(f"After  SMOTE: {X_balanced.shape[0]} samples")

# Plot after SMOTE
balanced_counts = pd.Series(y_balanced).value_counts().sort_index()
plt.figure(figsize=(10, 5))
bars2 = plt.bar(
    [class_names[i] for i in range(7)],
    [balanced_counts.get(i, 0) for i in range(7)],
    color=colors, edgecolor='black'
)
plt.title('Class Distribution After SMOTE (Balanced)', fontsize=14, fontweight='bold')
plt.xlabel('Skin Lesion Type')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
for bar, count in zip(bars2, [balanced_counts.get(i, 0) for i in range(7)]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 30,
             str(count), ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('02_after_smote.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_after_smote.png")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, random_state=42, stratify=y_balanced
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape}")
print(f"Test  set: {X_test.shape}")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("Saved: scaler.pkl")

# ============================================================
# STEP 4: KNN MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 4: KNN MODEL")
print("="*60)

start = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)
knn_time = time.time() - start

print(f"KNN Accuracy : {knn_acc*100:.2f}%")
print(f"Training Time: {knn_time:.1f}s")
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn,
      target_names=[class_names[i] for i in range(7)]))

joblib.dump(knn, 'knn_model.pkl')
print("Saved: knn_model.pkl")

# ============================================================
# STEP 5: RANDOM FOREST MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 5: RANDOM FOREST MODEL")
print("="*60)

start = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_time = time.time() - start

print(f"Random Forest Accuracy : {rf_acc*100:.2f}%")
print(f"Training Time          : {rf_time:.1f}s")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=[class_names[i] for i in range(7)]))

joblib.dump(rf, 'rf_model.pkl')
print("Saved: rf_model.pkl")

# ============================================================
# STEP 6: SVM MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 6: SVM MODEL")
print("="*60)
print("(This may take 5-10 minutes...)")

start = time.time()
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_time = time.time() - start

print(f"SVM Accuracy : {svm_acc*100:.2f}%")
print(f"Training Time: {svm_time:.1f}s")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm,
      target_names=[class_names[i] for i in range(7)]))

joblib.dump(svm, 'svm_model.pkl')
print("Saved: svm_model.pkl")

# ============================================================
# STEP 7: CONFUSION MATRICES
# ============================================================
print("\n" + "="*60)
print("STEP 7: CONFUSION MATRICES")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, y_pred, acc) in zip(axes, [
    ('KNN', y_pred_knn, knn_acc),
    ('Random Forest', y_pred_rf, rf_acc),
    ('SVM', y_pred_svm, svm_acc)
]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[class_names[i] for i in range(7)],
                yticklabels=[class_names[i] for i in range(7)])
    ax.set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('ML Models - Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_ml_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_ml_confusion_matrices.png")

# ============================================================
# STEP 8: ML COMPARISON CHART
# ============================================================
print("\n" + "="*60)
print("STEP 8: ML COMPARISON CHART")
print("="*60)

ml_models  = ['KNN', 'Random Forest', 'SVM']
ml_scores  = [knn_acc*100, rf_acc*100, svm_acc*100]
ml_colors  = ['#FF6B6B', '#4ECDC4', '#45B7D1']

plt.figure(figsize=(8, 5))
bars = plt.bar(ml_models, ml_scores, color=ml_colors, edgecolor='black', width=0.5)
plt.title('ML Models - Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
for bar, score in zip(bars, ml_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{score:.2f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('04_ml_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_ml_comparison.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("ML MODELS COMPLETE! 🎉")
print("="*60)
print(f"KNN           : {knn_acc*100:.2f}%")
print(f"Random Forest : {rf_acc*100:.2f}%")
print(f"SVM           : {svm_acc*100:.2f}%")
print("\nSaved Models:")
print("  knn_model.pkl")
print("  rf_model.pkl")
print("  svm_model.pkl")
print("  scaler.pkl")
print("\nSaved Graphs:")
print("  01_class_distribution.png")
print("  02_after_smote.png")
print("  03_ml_confusion_matrices.png")
print("  04_ml_comparison.png")
print("\nNext Step: Run 2_dl_cnn.py")