import streamlit as st
import pandas as pd
import pickle
import os
import ast

DATASET_DIR = "dataset"

st.set_page_config(page_title="Medicine Recommendation System", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f3f7ff;
        --card: #ffffff;
        --primary: #2f6feb;
        --primary-soft: #eaf2ff;
        --success: #1e9d73;
        --warning: #f5a623;
        --text: #11203a;
        --muted: #5d6b82;
        --border: #dfe9ff;
    }

    .stApp {
        background: linear-gradient(180deg, #eef5ff 0%, #f8fbff 100%);
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stTabs"] button {
        background: transparent;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px 12px 0 0;
        padding: 0.7rem 1rem;
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #2f6feb, #6aa6ff);
        color: white;
        border-color: transparent;
    }

    .kpi-box {
        background: linear-gradient(135deg, var(--primary-soft), #ffffff);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 20px rgba(47, 111, 235, 0.08);
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 10px 26px rgba(17, 32, 58, 0.04);
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .result-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
    }

    .pill {
        display: inline-block;
        background: #e8f7f2;
        color: var(--success);
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .info-list li {
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [value.strip()] if value.strip() else []
        except (ValueError, SyntaxError):
            cleaned = [item.strip() for item in value.split(",") if item.strip()]
            return cleaned if cleaned else [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


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

st.title("💊 Medical Recommendation Assistant")
st.caption("AI-powered symptom screening that suggests likely conditions and relevant care guidance.")

header_cols = st.columns([1.5, 1, 1, 1])
with header_cols[0]:
    st.markdown('<div class="pill">Clinical Decision Support</div>', unsafe_allow_html=True)
with header_cols[1]:
    st.markdown('<div class="kpi-box"><div style="font-size:0.75rem; color:#5d6b82;">Symptoms</div><div style="font-size:1.7rem; font-weight:800; color:#11203a;">%s</div></div>' % (len(symptom_cols)), unsafe_allow_html=True)
with header_cols[2]:
    st.markdown('<div class="kpi-box"><div style="font-size:0.75rem; color:#5d6b82;">Disease Classes</div><div style="font-size:1.7rem; font-weight:800; color:#11203a;">%s</div></div>' % (len(le.classes_)), unsafe_allow_html=True)
with header_cols[3]:
    st.markdown('<div class="kpi-box"><div style="font-size:0.75rem; color:#5d6b82;">Model</div><div style="font-size:1.25rem; font-weight:800; color:#11203a;">Random Forest</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🩺 Recommendation", "📊 Dataset", "📈 Model Info"])

# ---------------- TAB 1: Recommendation ----------------
with tab1:
    left_col, right_col = st.columns([1.35, 1.2])

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Select symptoms")
        selected_symptoms = st.multiselect(
            "Choose the symptoms that match the patient:",
            options=symptom_cols,
            help="Select all symptoms that are currently present.",
        )
        st.caption(f"Selected: {len(selected_symptoms)} symptom(s)")

        if st.button("Predict disease & recommendations", use_container_width=True):
            if not selected_symptoms:
                st.warning("Please select at least one symptom before predicting.")
            else:
                input_vector = [1 if col in selected_symptoms else 0 for col in symptom_cols]
                input_df = pd.DataFrame([input_vector], columns=symptom_cols)

                pred_encoded = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]
                confidence = max(pred_proba) * 100
                disease = le.inverse_transform([pred_encoded])[0]

                st.session_state["last_prediction"] = {
                    "disease": disease,
                    "confidence": confidence,
                    "symptoms": selected_symptoms,
                }
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if "last_prediction" in st.session_state:
            result = st.session_state["last_prediction"]
            st.markdown(f'<div class="pill">Likely condition</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">{result["disease"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-subtitle">Confidence: {result["confidence"]:.2f}%</div>', unsafe_allow_html=True)

            st.progress(min_value=0, max_value=100, value=result["confidence"])
            st.caption(f"Symptoms reviewed: {', '.join(result['symptoms']) if result['symptoms'] else 'None'}")

            desc_row = description[description["Disease"] == result["disease"]]
            if not desc_row.empty:
                st.write("**Description**")
                st.write(desc_row["Description"].values[0])

            med_row = medications[medications["Disease"] == result["disease"]]
            if not med_row.empty:
                meds = parse_list(med_row["Medication"].values[0])
                st.write("**Recommended Medicines**")
                st.markdown("<ul class='info-list'>" + "".join(f"<li>{m}</li>" for m in meds[:8]) + "</ul>", unsafe_allow_html=True)

            prec_row = precautions[precautions["Disease"] == result["disease"]]
            if not prec_row.empty:
                prec_cols = [c for c in precautions.columns if c.lower().startswith("precaution")]
                values = []
                for c in prec_cols:
                    val = prec_row[c].values[0]
                    if pd.notna(val):
                        values.append(str(val).strip())
                if values:
                    st.write("**Precautions**")
                    st.markdown("<ul class='info-list'>" + "".join(f"<li>{v}</li>" for v in values) + "</ul>", unsafe_allow_html=True)

            diet_row = diets[diets["Disease"] == result["disease"]]
            if not diet_row.empty:
                diet_items = parse_list(diet_row["Diet"].values[0])
                if diet_items:
                    st.write("**Suggested Diet**")
                    st.markdown("<ul class='info-list'>" + "".join(f"<li>{d}</li>" for d in diet_items) + "</ul>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-subtitle'>Your prediction result will appear here after selecting symptoms.</div>", unsafe_allow_html=True)
            st.write("")
            st.write("Tip: Choose the symptoms most closely related to the patient and then click the prediction button.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TAB 2: Dataset Preview ----------------
with tab2:
    st.subheader("Reference datasets")
    dataset_tabs = st.tabs(["Training data", "Medications", "Precautions", "Diets"])

    with dataset_tabs[0]:
        st.dataframe(training.head(20), use_container_width=True)
        st.caption(f"Shape: {training.shape}")

    with dataset_tabs[1]:
        st.dataframe(medications.head(15), use_container_width=True)

    with dataset_tabs[2]:
        st.dataframe(precautions.head(15), use_container_width=True)

    with dataset_tabs[3]:
        st.dataframe(diets.head(15), use_container_width=True)

# ---------------- TAB 3: Model Evaluation ----------------
with tab3:
    st.subheader("Model details")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown('<div class="section-card"><h4>Model</h4><p>Random Forest Classifier</p></div>', unsafe_allow_html=True)
    with info_col2:
        st.markdown(f'<div class="section-card"><h4>Symptoms</h4><p>{len(symptom_cols)} features</p></div>', unsafe_allow_html=True)
    with info_col3:
        st.markdown(f'<div class="section-card"><h4>Diseases</h4><p>{len(le.classes_)} classes</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("This system uses a trained Random Forest model to classify a patient’s symptoms into the most likely disease category. It then retrieves the matching description, treatment guidance, precautions, and suggested dietary advice from the dataset.")
    st.write("- Trained model artifact: model.pkl")
    st.write("- Label encoder: label_encoder.pkl")
    st.write("- Symptom feature list: symptom_columns.pkl")
    st.markdown('</div>', unsafe_allow_html=True)