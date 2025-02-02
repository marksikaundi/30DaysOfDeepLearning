# Required imports
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define input data model
class InputData(BaseModel):
    features: list

# Define prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    # Convert input to numpy array
    features = np.array(data.features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return {"prediction": prediction.tolist()}

# Define a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}