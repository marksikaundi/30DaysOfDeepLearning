from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Train a simple model
def train_model():
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model
    with open('iris_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

# Initialize FastAPI app
app = FastAPI()

# Load or train model
try:
    with open('iris_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    model = train_model()

# Define input data model
class IrisInput(BaseModel):
    features: list

# Define prediction endpoint
@app.post("/predict")
async def predict(data: IrisInput):
    try:
        # Convert input to numpy array
        features = np.array(data.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Get probability scores
        probabilities = model.predict_proba(features)

        return {
            "prediction": int(prediction[0]),
            "probability": probabilities[0].tolist()
        }
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}