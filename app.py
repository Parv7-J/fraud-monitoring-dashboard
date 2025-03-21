from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="Fraud Detection API")

# Security
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEYS = {"fraud-detection-key": "your-secret-key-123"}

DATABASE_URL = "sqlite:///fraud_db.sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

model_path = "xgboost_fraud_model.pk1"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Upload it to Colab.")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    fraud_source = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    payer_mobile: str
    payee_id: str
    transaction_channel: str

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]

class FraudReport(BaseModel):
    transaction_id: str
    is_fraud: bool

def apply_rules(transaction: dict) -> bool:
    """Apply custom business rules"""
    if transaction['amount'] > 10000:
        return True
    return False

@app.post("/predict", tags=["Real-time Detection"])
async def predict_fraud(
    transaction: TransactionRequest,
    api_key: str = Security(api_key_header)
):
    """Real-time fraud prediction endpoint"""
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API key")

    try:
        input_data = pd.DataFrame([transaction.dict()])

        input_data["transaction_channel"] = input_data["transaction_channel"].map({"w": 1, "mobile": 0})

        rule_fraud = apply_rules(transaction.dict())

        proba = model.predict_proba(input_data)[0][1]
        model_fraud = proba > 0.5

        is_fraud = rule_fraud or model_fraud
        fraud_source = "rule" if rule_fraud else "model" if model_fraud else "none"

        db = SessionLocal()
        try:
            db_transaction = Transaction(
                transaction_id=transaction.transaction_id,
                amount=transaction.amount,
                is_fraud=is_fraud,
                fraud_probability=proba,
                fraud_source=fraud_source
            )
            db.add(db_transaction)
            db.commit()
        finally:
            db.close()

        return {
            "transaction_id": transaction.transaction_id,
            "is_fraud": is_fraud,
            "fraud_probability": round(proba, 4),
            "fraud_source": fraud_source,
            "rules_triggered": rule_fraud
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/batch-predict", tags=["Batch Processing"])
async def batch_predict(
    batch_request: BatchRequest,
    api_key: str = Security(api_key_header)
):
    """Batch processing endpoint (synchronous)"""
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API key")

    results = []
    for transaction in batch_request.transactions:
        try:
            result = await predict_fraud(transaction, api_key)
            results.append(result)
        except Exception as e:
            results.append({"transaction_id": transaction.transaction_id, "error": str(e)})

    return {"results": results}

@app.post("/report-fraud", tags=["Fraud Reporting"])
async def report_fraud(
    report: FraudReport,
    api_key: str = Security(api_key_header)
):
    """Manually report fraud for ground truth collection"""
    if api_key not in API_KEYS.values():
        raise HTTPException(status_code=403, detail="Invalid API key")

    db = SessionLocal()
    try:
        transaction = db.query(Transaction).filter(
            Transaction.transaction_id == report.transaction_id
        ).first()

        if transaction:
            transaction.is_fraud = report.is_fraud
            db.commit()
            return {"status": "Updated"}

        raise HTTPException(status_code=404, detail="Transaction not found")

    finally:
        db.close()


with open("app.py", "w") as f:
    f.write("""
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="Fraud Detection API")

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
API_KEYS = {"fraud-detection-key": "your-secret-key-123"}

DATABASE_URL = "sqlite:///fraud_db.sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

model_path = "xgboost_fraud_model.pk1"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Upload it to Colab.")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    fraud_source = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    payer_mobile: str
    payee_id: str
    transaction_channel: str

@app.get("/")
def home():
    return {"message": "FastAPI is running in Colab!"}
""")


!ngrok config add-authtoken 2ucpbkn0BAWJtYiqg4j4Y0fhH9E_sJvfn5QUeEsCaChrjgsk

from pyngrok import ngrok

public_url = ngrok.connect(8000)
print("Public URL:", public_url)

!nohup uvicorn filename:app --host 0.0.0.0 --port 8000 &

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI with CORS enabled!"}
