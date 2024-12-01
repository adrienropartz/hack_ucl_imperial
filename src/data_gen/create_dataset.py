import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp


class ImageProcessor(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> Optional[List[float]]:
        pass


class MediaPipeHandProcessor(ImageProcessor):
    def __init__(self, min_detection_confidence: float = 0.5, max_num_hands: int = 1):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            min_detection_confidence=min_detection_confidence,
            max_num_hands=max_num_hands,
        )

    def process_image(self, image_path: str) -> Optional[List[float]]:
        data_aux = []
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_values = [lm.x for lm in hand_landmarks.landmark]
                y_values = [lm.y for lm in hand_landmarks.landmark]

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_values[i] - min(x_values))
                    data_aux.append(y_values[i] - min(y_values))

        return data_aux if data_aux else None


class DatasetBuilder:
    def __init__(self, processor: ImageProcessor):
        self.processor = processor
        self.data: List[List[float]] = []
        self.labels: List[str] = []

    def is_valid_label_dir(self, label_dir: str) -> bool:
        if not label_dir.isalpha() and label_dir != "UNKNOWN_LETTER":
            return False
        if not (not label_dir.isnumeric() and label_dir != "UNKNOWN_NUMBER"):
            return False
        return True

    def process_directory(self, data_dir: str) -> None:
        for label_dir in os.listdir(data_dir):
            if not self.is_valid_label_dir(label_dir):
                continue

            label_path = os.path.join(data_dir, label_dir)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                processed_data = self.processor.process_image(image_path)

                if processed_data:
                    self.data.append(processed_data)
                    self.labels.append(label_dir)


class DatasetSaver:
    @staticmethod
    def save_dataset(
        data: List[List[float]],
        labels: List[str],
        output_dir: str,
        filename: str = "data",
    ) -> None:
        with open(os.path.join(output_dir, f"{filename}.pickle"), "wb") as dataset:
            pickle.dump({"data": data, "labels": labels}, dataset)

    @staticmethod
    def print_unique_labels(labels: List[str]) -> None:
        unique_labels = sorted(set(labels), key=lambda item: (len(item), item))
        print(f"Labels: {unique_labels}")


def build_dataset():
    # Initialize components
    processor = MediaPipeHandProcessor()
    builder = DatasetBuilder(processor)

    # Process the data
    data_dir = "."
    builder.process_directory(data_dir)

    # Save the results
    saver = DatasetSaver()
    saver.save_dataset(builder.data, builder.labels, data_dir)
    saver.print_unique_labels(builder.labels)


if __name__ == "__main__":
    build_dataset()
