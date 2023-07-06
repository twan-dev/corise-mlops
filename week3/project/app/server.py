from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global news_classifier_global, log_global

    news_classifier_global = NewsCategoryClassifier()
    news_classifier_global.load(MODEL_PATH)

    log_global = open(LOGS_OUTPUT_PATH, "a+")
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    log_global.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    start_time = datetime.now()

    score = news_classifier_global.predict_proba(request)
    label = news_classifier_global.predict_label(request)
    response = PredictResponse(scores=score, label=label)

    log_data = {
        'timestamp': datetime.now().strftime('%Y:%m:%d %H:%M%S'),
        'request': request,
        'prediction': score,
        'latency': (datetime.now() - start_time).total_seconds() * 1000
    }

    log_global.write(str(log_data) + '\n')
    
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
