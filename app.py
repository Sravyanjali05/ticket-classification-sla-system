from datetime import datetime, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from fastapi import UploadFile, File
import pandas as pd


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = FastAPI()

class TicketInput(BaseModel):
    ticket_text: str

# load model
model = joblib.load("ticket_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text

def assign_priority(ticket_type: str) -> str:
    if ticket_type in ["Billing inquiry", "Refund request"]:
        return "High"
    elif ticket_type == "Technical issue":
        return "Medium"
    return "Low"

def get_sla_hours(priority: str) -> int:
    if priority == "High":
        return 4
    elif priority == "Medium":
        return 8
    return 24


def get_sla_status(created_time: datetime, priority: str) -> str:
    sla_hours = get_sla_hours(priority)
    deadline = created_time + timedelta(hours=sla_hours)
    now = datetime.now()

    if now > deadline:
        return "Breached"

    remaining = deadline - now
    if remaining.total_seconds() <= (0.25 * sla_hours * 3600):
        return "At Risk"

    return "On Track"

@app.post("/predict")
def predict_ticket(data: TicketInput):
    created_time = datetime.now()

    cleaned = clean_text(data.ticket_text)
    vector = tfidf.transform([cleaned])
    category = model.predict(vector)[0]
    priority = assign_priority(category)

    sla_status = get_sla_status(created_time, priority)

    return {
        "ticket_text": data.ticket_text,
        "predicted_category": category,
        "priority": priority,
        "created_time": created_time,
        "sla_status": sla_status
    }

@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if "ticket_text" not in df.columns:
        return {"error": "CSV must contain 'ticket_text' column"}

    results = []

    for text in df["ticket_text"]:
        created_time = datetime.now()

        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])
        category = model.predict(vector)[0]
        priority = assign_priority(category)
        sla_status = get_sla_status(created_time, priority)

        results.append({
            "ticket_text": text,
            "predicted_category": category,
            "priority": priority,
            "sla_status": sla_status
        })

    return results
