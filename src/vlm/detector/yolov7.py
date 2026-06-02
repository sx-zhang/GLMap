import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch

from vlm.coco_classes import COCO_CLASSES
from vlm.config import VLM_YOLOV7_HTTP_MODE, VLM_YOLOV7_PORT
from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

sys.path.insert(0, "third_party/yolov7/")
try:
    from models.experimental import attempt_load  # noqa: E402

    from utils.datasets import letterbox  # noqa: E402
    from utils.general import (  # noqa: E402
        check_img_size,
        non_max_suppression,
        scale_coords,
    )
    from utils.torch_utils import TracedModel  # noqa: E402
except (ModuleNotFoundError, ImportError):
    print("Could not import yolov7. This is OK if you are only using the client.")
sys.path.pop(0)


class YOLOv7:
    def __init__(
        self, weights: str, image_size: int = 640, half_precision: bool = True
    ):
        """Loads the model and saves it to a field."""
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.half_precision = self.device.type != "cpu" and half_precision
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(image_size, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # to FP16

        # Warm-up
        if self.device.type != "cpu":
            dummy_img = torch.rand(
                1, 3, int(self.image_size * 0.7), self.image_size
            ).to(self.device)
            if self.half_precision:
                dummy_img = dummy_img.half()
            for i in range(3):
                self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
        verbose: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        orig_shape = image.shape
        if verbose:
            print("yolov7 is detecting")
        # Preprocess image
        img = cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=cv2.INTER_AREA,
        )
        img = letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )[0]
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] /= orig_shape[1]
        pred[:, 1] /= orig_shape[0]
        pred[:, 2] /= orig_shape[1]
        pred[:, 3] /= orig_shape[0]
        boxes = pred[:, :4]
        logits = pred[:, 4]
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]
        detections = ObjectDetections(
            boxes, logits, phrases, image_source=image, fmt="xyxy"
        )
        return detections


class YOLOv7Client:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yolov7"

    def predict(
        self,
        image_numpy: np.ndarray,
        agnostic_nms: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            agnostic_nms=agnostic_nms,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


class YOLOv7Interface:
    def __init__(
        self,
        weights: str = "data/yolov7-e6e.pt",
        http_mode: bool = VLM_YOLOV7_HTTP_MODE,
        port: int = VLM_YOLOV7_PORT,
    ):
        self.http_mode = http_mode
        if self.http_mode:
            self.client = YOLOv7Client(port)
            self.model = None
        else:
            self.model = YOLOv7(weights)
            self.client = None

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        if self.http_mode:
            return self.client.predict(
                image_numpy=image,
                agnostic_nms=agnostic_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )
        else:
            # In local mode, ensure tensors are on CPU via to_json/from_json
            detections = self.model.predict(
                image,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                classes=classes,
                agnostic_nms=agnostic_nms,
            )
            # Use to_json and from_json to ensure tensors are correctly converted to CPU
            json_data = detections.to_json()
            return ObjectDetections.from_json(json_data, image_source=image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=VLM_YOLOV7_PORT)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOv7Server(ServerMixin, YOLOv7):
        def process_payload(self, payload: dict) -> dict:
            agnostic_nms = payload["agnostic_nms"]
            conf_thres = payload["conf_thres"]
            iou_thres = payload["iou_thres"]
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                agnostic_nms=agnostic_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                verbose=True,
            ).to_json()

    yolov7 = YOLOv7Server("data/yolov7-e6e.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov7, name="yolov7", port=args.port)
