from typing import List

from loguru import logger
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sentence_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    #  estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #  transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.pipeline = None
        self.classes = None

    def _initialize_pipeline(self) -> Pipeline:
        pipeline = Pipeline([
            ('transformer_featurizer', TransformerFeaturizer()),
            ('classifier', LogisticRegression(
                    multi_class='multinomial',
                    tol=0.001,
                    solver='saga',
            ))
        ], verbose=self.verbose)
        return pipeline

    def fit(self, X_train: List, y_train: List) -> None:
        logger.info("Beginning model training ...")
        if not self.pipeline:
            self.pipeline = self._initialize_pipeline()
        self.pipeline.fit(X_train, y_train)
        self.classes = self.pipeline['classifier'].classes_

    def dump(self, model_path: str) -> None:
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Saved trained model pipeline to: {model_path}")

    def load(self, model_path: str) -> None:
        logger.info(f"Loaded trained model pipeline from: {model_path}")
        self.pipeline = joblib.load(model_path)
        self.classes = self.pipeline['classifier'].classes_

    def predict_proba(self, model_input: dict) -> dict:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction probability scores across all labels

        Output format:
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2
            ...
        }
        """
        return dict(zip(self.classes, self.pipeline.predict_proba([model_input.description])[0]))

    def predict_label(self, model_input: dict) -> str:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        return self.pipeline.predict([model_input.description])
