# projector-test-task
## Test task for ML in Production course


**Stage 1. Training model**

Training can be reproduced with running following command:

```
pip install -r requirements.txt
python train.py -i path-to-train-data.csv -r save-reggressor-to.pkl  -m save-metrics-to.txt
```


**Stage 2. Implementing API Server with an endpoint for prediction**

Run using this command:
```
uvicorn api:app --reload
```
