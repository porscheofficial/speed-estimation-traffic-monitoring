from typing import List, Tuple

import cv2
import numpy as np
from paths import YOLOV4_WEIGHTS, YOLOV4_CLASSES, YOLOV4_CONFIG


class ObjectDetection:
    """This class is used to detect the cars in a frame."""

    def __init__(self, weights_path=YOLOV4_WEIGHTS, cfg_path=YOLOV4_CONFIG):
        """Create an instance of ObjectDetection.

        @param weights_path:
            The path to the model weights.
        @param cfg_path:
            The path to config file.
        """
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(
            size=(self.image_size, self.image_size), scale=1 / 255
        )

    def load_class_names(self, classes_path: str = YOLOV4_CLASSES) -> List[str]:
        """Get all classes the model can classify in an image.

        @param classes_path: The path to the classes.txt file
            (e.g., `speed_estimation/model_weights/classes.txt`).

        @return: Returns a list of all classes the model can detect.
        """
        with open(classes_path, "r", encoding="UTF-8") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))

        return self.classes

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Detect cars in frame and put bounding boxes around.

        This function detects all cars in the frame and puts 2D bounding boxes around.
        A YoloV4 model is used.

        @param frame:
            The frame that should be analyzed.

        @return:
            Returns a tuple of class_ids, scores, and boxes.
            The class_id identifies which object was detected, while the scores indicate the
            confidence level of the prediction.
            The boxes are represented as a list of NumPy ndarrays, where each array corresponds to
            a bounding box that identifies a car in the frame.
            Each ndarray in the list holds the following information:
            (x_coordinate, y_coordinate, width, height).
        """
        return self.model.detect(
            frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold
        )
