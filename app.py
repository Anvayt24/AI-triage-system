import streamlit as st
import requests
import json
from api import PatientData
from sqlalchemy import create_engine


translations = {
    "English": {
        "title": "Summer ER Triage System",
        "description": "Enter patient details to predict triage level (Green, Yellow, Red) based on summer symptoms.",
        "age": "Age (years)",
        "trestbps": "Blood Pressure (mmHg)",
        "chol": "Cholesterol (mg/dL)",
        "thalch": "Heart Rate (bpm)",
        "temp": "Temperature (°C)",
        "resp_rate": "Respiratory Rate (breaths/min)",
        "symptom_label": "Select Symptom",
        "predict_button": "Predict Triage",
        "result_header": "Triage Result",
        "red_message": "Triage Level: Red - Immediate attention required!",
        "yellow_message": "Triage Level: Yellow - Urgent but stable.",
        "green_message": "Triage Level: Green - Non-urgent.",
        "input_data": "Input Data:",
        "send_to_doctor": "Send to Doctor",
        "doctor_section": "Patient Data for Doctor",
        "instructions": """
            ### Instructions
            - Enter the patient's vital signs and select a symptom.
            - Click 'Predict Triage' to get the triage level.
            - Ensure the API is running (`uvicorn api:app --reload`).
        """,
        "symptoms": {
            "cough": "cough",
            "injury": "injury",
            "diarrhea": "diarrhea",
            "high fever": "high fever",
            "rash": "rash",
            "vomiting": "vomiting",
            "shortness of breath": "shortness of breath",
            "severe weakness": "severe weakness",
            "sweating excessively": "sweating excessively"
        }
    },
    "Hindi": {
        "title": "गर्मी आपातकालीन ट्राइएज सिस्टम",
        "description": "रोगी के विवरण दर्ज करें ताकि गर्मी के लक्षणों के आधार पर ट्राइएज स्तर (हरा, पीला, लाल) की भविष्यवाणी की जा सके।",
        "age": "आयु (वर्ष)",
        "trestbps": "रक्तचाप (मिमीएचजी)",
        "chol": "कोलेस्ट्रॉल (मिग्रा/डीएल)",
        "thalch": "हृदय गति (धड़कन/मिनट)",
        "temp": "तापमान (°से)",
        "resp_rate": "श्वसन दर (साँसें/मिनट)",
        "symptom_label": "लक्षण चुनें",
        "predict_button": "ट्राइएज की भविष्यवाणी करें",
        "result_header": "ट्राइएज परिणाम",
        "red_message": "ट्राइएज स्तर: लाल - तत्काल ध्यान आवश्यक!",
        "yellow_message": "ट्राइएज स्तर: पीला - तत्काल लेकिन स्थिर।",
        "green_message": "ट्राइएज स्तर: हरा - गैर-तत्काल।",
        "input_data": "इनपुट डेटा:",
        "send_to_doctor": "डॉक्टर को भेजें",
        "doctor_section": "डॉक्टर के लिए रोगी डेटा",
        "instructions": """
            ### निर्देश
            - रोगी के जीवन संकेत दर्ज करें और एक लक्षण चुनें।
            - ट्राइएज स्तर प्राप्त करने के लिए 'ट्राइएज की भविष्यवाणी करें' पर क्लिक करें।
            - सुनिश्चित करें कि API चल रहा है (`uvicorn api:app --reload`)।
        """,
        "symptoms": {
            "cough": "खांसी",
            "injury": "चोट",
            "diarrhea": "दस्त",
            "high fever": "तेज बुखार",
            "rash": "चकत्ते",
            "vomiting": "उल्टी",
            "shortness of breath": "सांस की तकलीफ",
            "severe weakness": "गंभीर कमजोरी",
            "sweating excessively": "अत्यधिक पसीना"
        }
    }
}    
language = st.selectbox("Select Language / भाषा चुनें ",
                       ["English", "Hindi"], key="language")
lang_dict = translations[language]

st.title(lang_dict["title"])
st.write(lang_dict["description"])


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    trestbps = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=120, step=1)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)

with col2:
    thalch = st.number_input("Heart Rate (bpm)", min_value=60, max_value=200, value=80, step=1)
    temp = st.number_input("Temperature (°C)", min_value=35.0, max_value=43.0, value=36.6, step=0.1)
    resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=40, value=16, step=1)


symptom_options = list(lang_dict["symptoms"].values())
symptom_map = {v: k for k, v in lang_dict["symptoms"].items()}  # Maps translated symptom back to English
selected_translated_symptom = st.selectbox(lang_dict["symptom_label"], symptom_options)
selected_symptom = symptom_map[selected_translated_symptom]


if st.button("Predict Triage"):
    patient_data = {
        "age": float(age),
        "trestbps": float(trestbps),
        "chol": float(chol),
        "thalch": float(thalch),
        "temp": float(temp),
        "resp_rate": float(resp_rate),
        "symptoms": selected_symptom
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=patient_data)
        response.raise_for_status()  # Check for HTTP errors
        result = response.json()

        # Display result
        triage = result["triage"]
        st.subheader("Triage Result")
        if triage == "Red":
            st.error(f"Triage Level: {triage} - Immediate attention required!")
        elif triage == "Yellow":
            st.warning(f"Triage Level: {triage} - Urgent but stable.")
        else:
            st.success(f"Triage Level: {triage} - Non-urgent.")
        
        st.write("Input Data:", result["input_data"])

        st.session_state["latest_patient"] = result

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")

if "latest_patient" in st.session_state and st.button(lang_dict["send_to_doctor"]):
    st.subheader(lang_dict["doctor_section"])
    patient = st.session_state["latest_patient"]
    triage = patient["triage"]
    data = patient["input_data"]
    st.write(f"**Triage Level**: {triage}")
    st.write(f"**Age**: {data['age']}")
    st.write(f"**Blood Pressure**: {data['trestbps']} mmHg")
    st.write(f"**Cholesterol**: {data['chol']} mg/dL")
    st.write(f"**Heart Rate**: {data['thalch']} bpm")
    st.write(f"**Temperature**: {data['temp']} °C")
    st.write(f"**Respiratory Rate**: {data['resp_rate']} breaths/min")
    st.write(f"**Symptoms**: {data['symptoms']}")
    st.success("Data sent to doctor (simulated). Check triage_log.db for records.")   

st.markdown(lang_dict["instructions"])