from fastapi import FastAPI
from pydantic import BaseModel, constr
from typing import List
from predictor import ScorePredictor

app = FastAPI()
predictor = ScorePredictor('./output/regressor.pkl')


class UserRequestIn(BaseModel):
    text: constr(min_length=1)


class ScoresOut(BaseModel):
    score: List[float]


@app.post("/score", response_model=ScoresOut)
def score_prediction(user_request_in: UserRequestIn):
    return predictor.predict_scores([user_request_in.text])