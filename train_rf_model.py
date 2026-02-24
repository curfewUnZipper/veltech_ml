import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("ev_12v_battery_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# FEATURES & LABEL
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# TRAIN TEST SPLIT (IMPORTANT)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print(f"\n🎯 Accuracy: {accuracy:.4f}")

# -----------------------------
# FEATURE IMPORTANCE (VERY USEFUL)
# -----------------------------
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n🔥 Feature Importance:")
print(importances)

# Optional plot
plt.figure(figsize=(6,4))
importances.plot(kind="bar")
plt.title("Feature Importance - Battery Failure Model")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance plot saved")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "battery_rf_model.pkl")
print("\n💾 Model saved as battery_rf_model.pkl")