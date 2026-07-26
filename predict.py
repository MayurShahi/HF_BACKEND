import joblib 
import pandas as pd


# Load the trained Model



model = joblib.load(r'C:\AI_Project\Heart_Failure_Prediction\app\random_forest.pkl')
scaler = joblib.load(r'C:\AI_Project\Heart_Failure_Prediction\models\scaler.pkl')

# Feature Order

FEATURE_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time"
]

def preprocess(data):
    """
    Preprocess the input data for prediction.
    
    Parameters:
    data (dict): Input data in dictionary format.
    
    Returns:
    pd.DataFrame: Preprocessed data ready for prediction.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Ensure the DataFrame has all required columns
    df = df[FEATURE_COLUMNS]
    
    # Standardize 
    df = pd.DataFrame(
    scaler.transform(df),
    columns=FEATURE_COLUMNS
    )
    
    return df

def predict(data):
    """
    Make a prediction using the trained model.
    
    Parameters:
    data (dict): Input data in dictionary format.
    
    Returns:
    dict: Prediction result with the probability of heart failure.
    """
    # Preprocess the input data
    preprocessed_data = preprocess(data)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)[0]
    probability = model.predict_proba(preprocessed_data)[0]
    
    return {
        "prediction": int(prediction),
    "probability_negative": round(float(probability[0]), 4),
    "probability_positive": round(float(probability[1]), 4)
    }

if __name__ == "__main__":

    sample = {
        "age": 60,
        "anaemia": 0,
        "creatinine_phosphokinase": 582,
        "diabetes": 0,
        "ejection_fraction": 20,
        "high_blood_pressure": 1,
        "platelets": 265000,
        "serum_creatinine": 1.9,
        "serum_sodium": 130,
        "sex": 1,
        "smoking": 0,
        "time": 4
    }

    result = predict(sample)

    print(result)

