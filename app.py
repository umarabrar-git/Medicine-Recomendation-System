import streamlit as st
import pandas as pd
import pickle
import os
import ast

DATASET_DIR = "dataset"

st.set_page_config(page_title="Medicine Recommendation System", layout="wide")

# ---- Load model and helper files ----
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("symptom_columns.pkl", "rb") as f:
        symptom_cols = pickle.load(f)
    return model, le, symptom_cols

model, le, symptom_cols = load_artifacts()

@st.cache_data
def load_reference_data():
    description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
    precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
    medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
    diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))
    training = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))
    return description, precautions, medications, diets, training

description, precautions, medications, diets, training = load_reference_data()

st.title("💊 Medicine Recommendation System")

tab1, tab2, tab3 = st.tabs(["🩺 Get Recommendation", "📊 Dataset Preview", "📈 Model Evaluation"])

# ---------------- TAB 1: Recommendation ----------------
with tab1:
    st.subheader("Enter Your Symptoms")
    selected_symptoms = st.multiselect(
        "Select symptoms you are experiencing:",
        options=symptom_cols
    )

    if st.button("Predict Disease & Get Recommendation"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Build input vector
            input_vector = [1 if col in selected_symptoms else 0 for col in symptom_cols]
            input_df = pd.DataFrame([input_vector], columns=symptom_cols)

            # Predict
            pred_encoded = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            confidence = max(pred_proba) * 100
            disease = le.inverse_transform([pred_encoded])[0]

            st.success(f"**Predicted Condition:** {disease}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # ---- Description ----
            desc_row = description[description["Disease"] == disease]
            if not desc_row.empty:
                st.write("**Description:**", desc_row["Description"].values[0])

            # ---- Medications ----
            med_row = medications[medications["Disease"] == disease]
            if not med_row.empty:
                raw_meds = med_row["Medication"].values[0]
                try:
                    med_list = ast.literal_eval(raw_meds)
                except (ValueError, SyntaxError):
                    med_list = [raw_meds]
                st.write("**Recommended Medicines:**")
                for m in med_list:
                    st.write(f"- {m}")

            # ---- Precautions ----
            prec_row = precautions[precautions["Disease"] == disease]
            if not prec_row.empty:
                prec_cols = [c for c in precautions.columns if c.lower().startswith("precaution")]
                st.write("**Precautions:**")
                for c in prec_cols:
                    val = prec_row[c].values[0]
                    if pd.notna(val):
                        st.write(f"- {val}")

            # ---- Diet ----
            diet_row = diets[diets["Disease"] == disease]
            if not diet_row.empty:
                raw_diet = diet_row["Diet"].values[0]
                try:
                    diet_list = ast.literal_eval(raw_diet)
                except (ValueError, SyntaxError):
                    diet_list = [raw_diet]
                st.write("**Suggested Diet:**")
                for d in diet_list:
                    st.write(f"- {d}")

# ---------------- TAB 2: Dataset Preview ----------------
with tab2:
    st.subheader("Training Dataset Sample")
    st.dataframe(training.head(20))
    st.write("Shape:", training.shape)

    st.subheader("Medications Sample")
    st.dataframe(medications.head(10))

    st.subheader("Precautions Sample")
    st.dataframe(precautions.head(10))

    st.subheader("Diets Sample")
    st.dataframe(diets.head(10))

# ---------------- TAB 3: Model Evaluation ----------------
with tab3:
    st.subheader("Model Info")
    st.write("Model type: Random Forest Classifier")
    st.write("Number of symptom features:", len(symptom_cols))
    st.write("Number of disease classes:", len(le.classes_))
    st.write("Trained model loaded from model.pkl")