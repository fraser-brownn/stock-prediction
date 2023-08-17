from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from typing import NewType, Optional
import pandas as pd

def load_model(ticker):
    model = load(pathlib.Path(f'./models/saved/{ticker}_model.joblib'))
    return model

app = FastAPI(title = 'Stock Prediction')


class InputData(BaseModel):
    open:float      
    high:float       
    low:float     
    close:float    
    volume:int
    RSI:float
    ma_3:float
    ma_10:float

class OutputData(BaseModel):
    score: float


@app.post('/score/{ticker}', response_model = OutputData)
async def score(data:InputData, ticker:str):
    model = load_model(ticker)
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)
    return OutputData(score = result)


