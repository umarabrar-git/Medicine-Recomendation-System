import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---- Load cleaned data ----
X = pd.read_csv("X_clean.csv")
y = pd.read_csv("y_clean.csv").squeeze()

# ---- Encode labels ----
le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ---- Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ---- Train RandomForest ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

labels_present = np.unique(np.concatenate([y_test, y_pred]))
print(classification_report(
    y_test, y_pred,
    labels=labels_present,
    target_names=le.classes_[labels_present]
))

# ---- Save model ----
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")