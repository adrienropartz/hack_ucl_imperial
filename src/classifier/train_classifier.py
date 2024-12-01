import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    @staticmethod
    def load_data(data_path: str) -> Dict:
        with open(data_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def clean_data(data_dict: Dict, expected_dimension: int = 42) -> Dict:
        indices_to_remove = []
        for i, val in enumerate(data_dict["data"]):
            if len(val) != expected_dimension:
                print(
                    f"Dimension: {len(val)}, line: {i}, Class: {data_dict['labels'][i]}"
                )
                indices_to_remove.append(i)

        for index in sorted(indices_to_remove, reverse=True):
            del data_dict["data"][index]
            del data_dict["labels"][index]

        return data_dict

    @staticmethod
    def prepare_training_data(data_dict: Dict, test_size: float = 0.2):
        data = np.asarray(data_dict["data"])
        labels = np.asarray(data_dict["labels"])

        return train_test_split(
            data, labels, test_size=test_size, shuffle=True, stratify=labels
        )


class ModelSaver:
    @staticmethod
    def save_model(model: BaseEstimator, model_path: str) -> None:
        with open(model_path, "wb") as model_file:
            pickle.dump({"model": model}, model_file)


class ModelTrainer(ABC):
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        pass


class RandomForestTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return self.model.predict(x_test)

    @property
    def model_instance(self) -> BaseEstimator:
        return self.model


class ModelEvaluator:
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        score = accuracy_score(y_true, y_pred)
        print(f"{score:.2%} of samples were classified correctly!")
        return score

    @staticmethod
    def get_classification_report(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> str | Dict[Any, Any]:
        report = classification_report(y_true, y_pred, digits=4)
        print(f"Classification Report:\n{report}\n")
        return report

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, labels: List, ax: plt.Axes
    ) -> None:
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
        display.plot(cmap=plt.cm.Blues, values_format="g", ax=ax)
        plt.title("Confusion Matrix")
        plt.show()


def main():
    # Configuration
    data_path = "../data/data.pickle"
    model_path = "./model.pickle"

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = RandomForestTrainer()
    evaluator = ModelEvaluator()

    # Data preparation
    data_dict = preprocessor.load_data(data_path)
    clean_data = preprocessor.clean_data(data_dict)
    x_train, x_test, y_train, y_test = preprocessor.prepare_training_data(clean_data)

    # Model training and evaluation
    trainer.train(x_train, y_train)
    y_pred = trainer.predict(x_test)

    # Evaluation
    evaluator.calculate_accuracy(y_test, y_pred)
    evaluator.get_classification_report(y_test, y_pred)

    # Optional: Plot confusion matrix
    _, ax = plt.subplots(figsize=(8, 6))
    evaluator.plot_confusion_matrix(y_test, y_pred, list(set(y_test)), ax)

    # Save model
    ModelSaver.save_model(trainer.model_instance, model_path)


if __name__ == "__main__":
    main()
