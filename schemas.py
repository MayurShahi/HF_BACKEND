from pydantic import BaseModel, Field

class HeartFailureInput(BaseModel):
    age: float = Field(..., example=60)
    anaemia: int = Field(..., ge=0, le=1, example=0)
    creatinine_phosphokinase: float = Field(..., example=582)
    diabetes: int = Field(..., ge=0, le=1, example=0)
    ejection_fraction: float = Field(..., example=20)
    high_blood_pressure: int = Field(..., ge=0, le=1, example=1)
    platelets: float = Field(..., example=265000)
    serum_creatinine: float = Field(..., example=1.9)
    serum_sodium: float = Field(..., example=130)
    sex: int = Field(..., ge=0, le=1, example=1)
    smoking: int = Field(..., ge=0, le=1, example=0)
    time: float = Field(..., example=4)


class PredictionResponse(BaseModel):
    prediction: int
    probability_negative: float
    probability_positive: float

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str


class PredictionExplanationRequest(BaseModel):
    message: str
    patient_data: HeartFailureInput


class PredictionExplanationResponse(BaseModel):
    prediction: int
    probability_negative: float
    probability_positive: float
    explanation: str