"""
Wine Quality Prediction
Project 2 Proposal - Level 2
Dataset: WineQT.csv (1,143 rows, red Vinho Verde wine variant)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════
df = pd.read_csv('WineQT.csv')
print(f"Shape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")

df.drop(columns=['Id'], inplace=True)

# ══════════════════════════════════════════════════════════
# 2. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════
print(f"\nCorrelation with quality:\n{df.corr()['quality'].sort_values(ascending=False)}")

# ══════════════════════════════════════════════════════════
# 3. DATA PREPARATION
# ══════════════════════════════════════════════════════════
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ══════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# Stochastic Gradient Descent
sgd = SGDClassifier(loss='log_loss', max_iter=2000, random_state=42)
sgd.fit(X_train_scaled, y_train)
sgd_pred = sgd.predict(X_test_scaled)

# Support Vector Classifier
svc = SVC(kernel='rbf', C=1.5, gamma='scale', random_state=42)
svc.fit(X_train_scaled, y_train)
svc_pred = svc.predict(X_test_scaled)

# ══════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════
for name, pred in [('Random Forest', rf_pred), ('SGD Classifier', sgd_pred), ('SVC', svc_pred)]:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"\n{name} - Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, pred, zero_division=0))

# Feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nFeature Importances (Random Forest):\n{importances}")
