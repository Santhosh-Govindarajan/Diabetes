import streamlit as st # type: ignore
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Diabetes Prediction App")

st.write("Please enter the following information:")

# Gender and Yes/No maps
gender_options = {"Male": 1, "Female": 0, "Other": 2}
yes_no_map = {"No": 0, "Yes": 1}

# Smoking history options sorted like LabelEncoder (capital first, then alphabetical)
smoking_labels = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
smoking_options = {label: idx for idx, label in enumerate(smoking_labels)}

# Input fields
gender = st.selectbox("Gender", list(gender_options.keys()))
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", list(yes_no_map.keys()))
heart_disease = st.selectbox("Heart Disease", list(yes_no_map.keys()))
smoking_history = st.selectbox("Smoking History", smoking_labels)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=0, max_value=300, value=100)

# Prepare input
input_data = np.array([[
    gender_options[gender],
    age,
    yes_no_map[hypertension],
    yes_no_map[heart_disease],
    smoking_options[smoking_history],
    bmi,
    hba1c,
    glucose
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The model predicts that the user has Diabetes.")
    else:
        st.success("The model predicts that the user does NOT have Diabetes.")
