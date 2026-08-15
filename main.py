from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import predict
from chatbot import medical_chat
from schemas import (
    HeartFailureInput,
    PredictionResponse,
    ChatRequest,
    
    ChatResponse,
    PredictionExplanationRequest,
    PredictionExplanationResponse
)

app = FastAPI(
    title = "Heart Failure Prediction API",
    description = "An API for predicting heart failure and providing medical advice.",
    version = "1.0.0"
)
# Enable CORS ( adjust origins for prodcution )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return{
        "message" : "Heart failure Prediction API is running."
    }

@app.get("/health")
def health():
    return {
        "status" : "healthy"
    }

@app.post(
    "/predict",
    response_model = PredictionResponse
)

def predict_endpoint(data: HeartFailureInput):

    try:
        result = predict (data.model_dump())
        return result

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )
    
@app.post(
    "/chat",
    response_model = ChatResponse
)
def chat_endpoint( request: ChatRequest):

    try:
        response = medical_chat(
            user_message = request.message
        )

        return{
            "response" : response
        }
    
    except Exception as e:

        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )

@app.post(
    "/chat/prediction",
    response_model = PredictionExplanationResponse
    )

def explain_prediction( request : PredictionExplanationRequest):

    try:
        #Convert the Pydantic model to a dictionary
        patient_data = request.patient_data.model_dump()

        # Run the Model 
        prediction_result = predict(patient_data)

        # ask the chatbot to explain the prediction
        explanation = medical_chat(
            user_message = request.message,
            prediction_result = prediction_result,
            patient_data = patient_data
        )

        #Return both prediction and explanation
        return{
            "prediction": prediction_result["prediction"],
            "probability_negative" : prediction_result["probability_negative"],
            "probability_positive" : prediction_result["probability_positive"],
            "explanation": explanation

        }
    
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail = str(e)
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)