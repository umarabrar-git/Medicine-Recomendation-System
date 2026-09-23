import pandas as pd
import numpy as np
import pickle
import os

DATASET_DIR = "dataset"

# ---- Load datasets ----
training = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))
description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))

print("Training shape:", training.shape)
print(training.head())

# ---- Clean ----
# Drop unnamed index columns if present
training = training.loc[:, ~training.columns.str.contains("^Unnamed")]

# Drop duplicates
training = training.drop_duplicates()

# Drop rows with missing target
training = training.dropna(subset=["prognosis"])

# Fill any missing symptom values with 0 (not present)
symptom_cols = [c for c in training.columns if c != "prognosis"]
training[symptom_cols] = training[symptom_cols].fillna(0)

# ---- Features and labels ----
X = training[symptom_cols]
y = training["prognosis"]

# Save the list of symptom column names (needed later in the app)
with open("symptom_columns.pkl", "wb") as f:
    pickle.dump(list(symptom_cols), f)

# Save cleaned data for training script
X.to_csv("X_clean.csv", index=False)
y.to_csv("y_clean.csv", index=False)

print("Preprocessing done. Cleaned files saved: X_clean.csv, y_clean.csv, symptom_columns.pkl")