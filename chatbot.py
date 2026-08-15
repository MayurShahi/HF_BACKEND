from groq import Groq

from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT ="""
You are an AI medical assistant integrated into a Heart Failure Prediction System.

Responsibilities:
- Explain heart failure in simple language.
- Explain machine learning prediction results.
- Explain medical terms.
- Suggest healthy lifestyle habits.
- Encourage users to seek professional medical advice.

Never:
- Diagnose diseases.
- Prescribe medications.
- Replace a doctor.

Always mention that your responses are educational and not a substitute for professional medical advice.
"""

def build_context(prediction_result: dict, patient_data: dict) -> str:
    return f"""
Prediction Result
-----------------
Prediction: {prediction_result["prediction"]}
Probability of Heart Failure: {prediction_result["probability_positive"]:.2%}
Probability of No Heart Failure: {prediction_result["probability_negative"]:.2%}

Patient Information
-------------------
Age: {patient_data["age"]}
Anaemia: {"Yes" if patient_data["anaemia"] else "No"}
Creatinine Phosphokinase: {patient_data["creatinine_phosphokinase"]}
Diabetes: {"Yes" if patient_data["diabetes"] else "No"}
Ejection Fraction: {patient_data["ejection_fraction"]}%
High Blood Pressure: {"Yes" if patient_data["high_blood_pressure"] else "No"}
Platelets: {patient_data["platelets"]}
Serum Creatinine: {patient_data["serum_creatinine"]}
Serum Sodium: {patient_data["serum_sodium"]}
Sex: {"Male" if patient_data["sex"] else "Female"}
Smoking: {"Yes" if patient_data["smoking"] else "No"}
Follow-up Time: {patient_data["time"]} days
"""

def medical_chat(
    user_message: str,
    prediction_result: dict | None = None,
    patient_data: dict | None = None
):
    context = ""

    if prediction_result is not None and patient_data is not None:
        context = build_context(prediction_result, patient_data)

    completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b-chat",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"{context}\n\nQuestion:\n{user_message}"
            }
        ],
        temperature=0.4,
        max_tokens=500
    )

    return completion.choices[0].message.content
