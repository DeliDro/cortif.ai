from pandas import DataFrame
from joblib import load

class MODELS:
    LOGISTIC_REGRESSION = "logreg"
    DECISION_TREE = "tree"
    NN = "nn"

class InputFeatures:
    def __init__(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def to_df(self):
        return (
            DataFrame([{
                "sepal_length": self.sepal_length,
                "sepal_width": self.sepal_width,
                "petal_length": self.petal_length,
                "petal_width": self.petal_width
            }])
        )

class Inferer:
    def __init__(self, model: MODELS):
        self.model = load(f"models/{model}/model.joblib")
        self.scaler = load(f"models/scaler.joblib")

    def predict(self, input_features: InputFeatures):
        try:
            transformed_input = self.scaler.transform(input_features.to_df())
            prediction = self.model.predict(transformed_input)
            return prediction[0]
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")

if __name__ == "__main__":
    input_obj = InputFeatures(7, 3.2, 4.7, 1.4)
    inferer = Inferer(MODELS.LOGISTIC_REGRESSION)
    print(inferer.predict(input_obj))
