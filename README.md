# 💊 Medicine Recommendation System

A **Machine Learning-based Medicine Recommendation System** that predicts suitable medicines based on the user's entered **symptoms, signs, and disease information**.

The project provides a simple and interactive web interface using **Streamlit**, allowing users to enter their health-related information and receive a model-generated medicine recommendation.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Users should consult a qualified healthcare professional before taking any medicine.

---

## 📌 Project Overview

The Medicine Recommendation System uses machine learning techniques to analyze user-provided medical information such as:

* Symptoms
* Signs
* Disease/condition

Based on these inputs, the trained machine learning model predicts a medicine recommendation.

The application is deployed through a **Streamlit web interface**, making it easy for users to interact with the trained model without requiring programming knowledge.

---

## ✨ Features

* 🩺 Enter symptoms and signs
* 🦠 Enter or select a disease
* 🤖 Machine learning-based prediction
* 💊 Medicine recommendation
* 🌐 Interactive Streamlit interface
* ⚡ Fast prediction
* 📊 Simple and user-friendly UI
* 🔄 Easy to extend with additional diseases, symptoms, and medicines

---

## 🏗️ System Workflow

```text
                User
                  │
                  ▼
        ┌───────────────────┐
        │ Enter Symptoms,   │
        │ Signs & Disease   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Data Preprocessing│
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Machine Learning  │
        │      Model        │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Medicine          │
        │ Prediction        │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Streamlit UI      │
        │ Display Result    │
        └───────────────────┘
```

---

## 🧠 Machine Learning

The system uses a trained machine learning model to learn relationships between:

**Symptoms + Signs + Disease → Medicine**

The general machine learning pipeline consists of:

1. Data collection
2. Data preprocessing
3. Feature selection
4. Data transformation
5. Model training
6. Model evaluation
7. Model saving
8. Prediction through Streamlit

The exact algorithm can be specified according to the model used in the project.

---

## 🛠️ Technologies Used

| Technology         | Purpose                           |
| ------------------ | --------------------------------- |
| Python             | Programming language              |
| Pandas             | Data manipulation                 |
| NumPy              | Numerical operations              |
| Scikit-learn       | Machine learning                  |
| Streamlit          | Web application                   |
| Pickle/Joblib      | Saving and loading trained models |
| Matplotlib/Seaborn | Data visualization, if used       |

---

## 📂 Project Structure

```text
Medicine-Recommendation-System/
│
├── app.py
├── model/
│   └── medicine_model.pkl
│
├── data/
│   └── medicine_dataset.csv
│
├── notebooks/
│   └── medicine_prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The folder structure can be modified according to the actual files in the project.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/umarabrar-git/medicine-recommendation-system.git
```

### 2. Navigate to the Project

```bash
cd medicine-recommendation-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this URL in your browser.

---

## 💻 How to Use

### Step 1

Open the Streamlit application.

### Step 2

Enter the patient's symptoms.

Example:

```text
fever, headache, body pain
```

### Step 3

Enter the relevant signs.

Example:

```text
high temperature, weakness
```

### Step 4

Enter or select the disease/condition.

Example:

```text
Flu
```

### Step 5

Click the **Predict Medicine** button.

### Step 6

The trained model processes the input and displays the predicted medicine.

---

## 📊 Example

### Input

```text
Symptoms:
Fever, headache, body pain

Signs:
High temperature, weakness

Disease:
Flu
```

### Output

```text
Predicted Medicine:
[Model Prediction]
```

The actual prediction depends on the trained model and dataset.

---

## 📈 Future Improvements

The project can be extended with:

* Multiple medicine recommendations
* Medicine dosage information
* Medicine side-effect information
* Drug interaction checking
* Patient history
* User authentication
* Doctor consultation integration
* Voice-based symptom input
* Multilingual support
* Explainable AI
* Model confidence/probability
* Cloud deployment
* Mobile application
* Larger and clinically validated datasets

---

## ⚠️ Medical Disclaimer

This application is an **educational machine learning project** and should not be used for self-diagnosis, self-medication, or emergency medical decisions.

Machine learning predictions may be incorrect or incomplete. A medicine recommendation generated by this system should **not** be considered a medical prescription.

Always consult a qualified doctor or healthcare professional before starting, stopping, or changing any medication.

---

## 👨‍💻 Author

**Umar Abrar**

BS Data Science
Machine Learning & Data Science Project

---

## 📄 License

This project is intended for educational and research purposes. Add an appropriate open-source license, such as the MIT License, if you plan to distribute the project publicly.
