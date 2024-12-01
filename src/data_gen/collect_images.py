import os
from abc import ABC, abstractmethod
import cv2
from typing import List, Tuple

class ImageCapture(ABC):
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def capture_frame(self) -> Tuple[bool, cv2.Mat]:
        pass
    
    @abstractmethod
    def release(self):
        pass

class WebcamCapture(ImageCapture):
    def __init__(self, height: int = 600, width: int = 800, fps: int = 30):
        self.height = height
        self.width = width
        self.fps = fps
        self.capture = None

    def initialize(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def capture_frame(self) -> Tuple[bool, cv2.Mat]:
        success, frame = self.capture.read()
        return success, cv2.flip(frame, 1) if success else None

    def release(self):
        if self.capture:
            self.capture.release()

class ImageStorage(ABC):
    @abstractmethod
    def save_image(self, image: cv2.Mat, path: str):
        pass

class LocalImageStorage(ImageStorage):
    def save_image(self, image: cv2.Mat, path: str):
        cv2.imwrite(path, image)

class ImageDisplay:
    GREEN = (0, 255, 0)

    @staticmethod
    def show_ready_screen(frame: cv2.Mat) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Press "R" to start'
        textsize = cv2.getTextSize(text, font, 1.3, 3)[0]
        
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = (frame.shape[0] + textsize[1]) // 2
        
        cv2.putText(frame, text, (textX, textY), font, 1.3, ImageDisplay.GREEN, 3, cv2.LINE_AA)
        cv2.imshow("frame", frame)

    @staticmethod
    def show_frame(frame: cv2.Mat) -> None:
        cv2.imshow("frame", frame)

class DataCollector:
    def __init__(self, 
                 capture_device: ImageCapture,
                 storage: ImageStorage,
                 display: ImageDisplay,
                 data_dir: str = "."):
        self.capture_device = capture_device
        self.storage = storage
        self.display = display
        self.data_dir = data_dir

    def collect_images(self, class_labels: List[str], dataset_size: int = 100) -> None:
        self.capture_device.initialize()

        try:
            for label in class_labels:
                self._collect_class_images(label, dataset_size)
        finally:
            self.capture_device.release()
            cv2.destroyAllWindows()

    def _collect_class_images(self, label: str, dataset_size: int) -> None:
        label_path = os.path.join(self.data_dir, label.upper())
        os.makedirs(label_path, exist_ok=True)
        print(f"Collecting data for class {label.upper()}")

        self._wait_for_ready()
        
        for counter in range(dataset_size):
            success, frame = self.capture_device.capture_frame()
            if success:
                self.display.show_frame(frame)
                cv2.waitKey(10)
                self.storage.save_image(
                    frame,
                    os.path.join(label_path, f"{100+counter}.jpg")
                )

    def _wait_for_ready(self) -> None:
        while True:
            success, frame = self.capture_device.capture_frame()
            if success:
                self.display.show_ready_screen(frame)
                if cv2.waitKey(10) == ord("r"):
                    break

if __name__ == "__main__":
    classes = ["M", "N"]
    dataset_size = 200
    
    collector = DataCollector(
        capture_device=WebcamCapture(),
        storage=LocalImageStorage(),
        display=ImageDisplay()
    )
    collector.collect_images(classes, dataset_size)

