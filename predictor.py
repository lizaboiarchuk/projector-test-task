import pickle
import tensorflow_hub as hub


class ScorePredictor:
    def __init__(self, regressor_path):
        self.encoder_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.encoder = hub.load(self.encoder_url)
        with open(regressor_path, 'rb') as f:
            self.regressor = pickle.load(f)

    def predict_scores(self, texts):
        vectors = self.encoder(texts).numpy()
        scores = self.regressor.predict(vectors)
        return {"score": list(scores)}
