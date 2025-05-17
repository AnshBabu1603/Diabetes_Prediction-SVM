from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Add CORS middleware here
origins = [
    "https://diabetespredictor-ruby.vercel.app/",  # your frontend domain
    # you can add other allowed origins or use ["*"] to allow all
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins, but better to be specific
    allow_credentials=True,
    allow_methods=["*"],    # allow all HTTP methods (GET, POST, OPTIONS, etc)
    allow_headers=["*"],    # allow all headers
)

model = joblib.load('SVM.pkl')
scaler = joblib.load('Scaler.pkl')

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post('/predict')
def predict(data: PatientData):
    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]])

    input_scaled = scaler.transform(input_data)
    prediction_prob = model.predict_proba(input_scaled)[0,1]
    prediction = model.predict(input_scaled)[0]

    return {
        'prediction': int(prediction),
        'probability': float(prediction_prob)
    }
