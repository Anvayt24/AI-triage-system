from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

app = FastAPI()
model = load_model('triage_model_refined2.h5')  # Use your latest model
scaler = joblib.load('scaler_refined2.pkl')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

engine = create_engine('sqlite:///triage_data.db' , echo=True)
Base = declarative_base()

class triagelog(Base):
    __tablename__ = 'triagelog'
    id = Column(Integer, primary_key=True , autoincrement=True)
    age = Column(Float)
    trestbps = Column(Float)
    chol = Column(Float)
    thalch = Column(Float)
    temp = Column(Float)
    resp_rate = Column(Float)
    symptoms = Column(String)
    triage = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class PatientData(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalch: float
    temp: float
    resp_rate: float
    symptoms: str

numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'temp','resp_rate']
symptom_cols = ['symptoms_cough', 'symptoms_injury', 'symptoms_diarrhea', 'symptoms_high fever', 
                'symptoms_rash', 'symptoms_vomiting', 'symptoms_shortness of breath', 
                'symptoms_severe weakness', 'symptoms_sweating excessively']

symptom_map = {s.split('_')[1]: i for i, s in enumerate(symptom_cols)}
valid_symptoms = list(symptom_map.keys())
symptom_aliases = {
    'coughing': 'cough',  # Example aliases for common misspellings or variations
    'fever': 'high fever',
    'breathing difficulty': 'shortness of breath',
    'weakness': 'severe weakness',
    'sweating': 'sweating excessively'
}

@app.post("/predict")
async def predict(data: PatientData):
    data_dict = data.dict()
    print("Input:", data_dict)
    symptoms_input = data_dict['symptoms'].lower().strip()
    symptoms_input = symptom_aliases.get(symptoms_input, symptoms_input)
    print("Processed symptoms:", symptoms_input)  # Should be 'snake bite'

    """if symptoms_input == 'snake bite' or symptoms_input == 'bleeding' or 
       data_dict['trestbps'] < 90 or data_dict['temp'] > 40 or data_dict['thalch'] > 140:
        print("Rule-based override: Red")
        return {"triage": "Red", "input_data": data_dict}
    
    if symptoms_input not in valid_symptoms:
        raise HTTPException(status_code=400, detail=f"Invalid symptom: '{symptoms_input}'. Must be one of {valid_symptoms}")"""
    
    input_df = pd.DataFrame([data_dict], columns=numeric_features + ['symptoms'])
    numeric_input = scaler.transform(input_df[numeric_features])
    categorical_input = np.zeros((1, len(symptom_cols)))
    symptom_index = symptom_map[symptoms_input]
    print("Symptom index:", symptom_index)  # Should be 6
    categorical_input[0, symptom_index] = 1
    processed_input = np.hstack((numeric_input, categorical_input))
    print("Model input:", processed_input)
    
    prediction = model.predict(processed_input)
    print("Prediction probs:", prediction)
    triage_class = np.argmax(prediction, axis=1)[0]
    triage_label = ['Green', 'Yellow', 'Red'][triage_class]
    print("Predicted:", triage_label)

    session = Session()
    log_entry = triagelog(
        age=data_dict['age'],
        trestbps=data_dict['trestbps'],
        chol=data_dict['chol'],
        thalch=data_dict['thalch'],
        temp=data_dict['temp'],
        resp_rate=data_dict['resp_rate'],
        symptoms=symptoms_input,
        triage=triage_label
    )
    session.add(log_entry)
    session.commit()
    session.close()
    
    return {"triage": triage_label, "input_data": data_dict}



@app.get("/")
async def root():
    return RedirectResponse(url="/docs")