import sys
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlm.config import VLM_GROUNDING_DINO_HTTP_MODE, VLM_GROUNDING_DINO_PORT
from vlm.detector.detections import ObjectDetections

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

sys.path.insert(0, "third_party/GroundingDINO/")
try:
    import groundingdino.datasets.transforms as T  # noqa: E402
    from groundingdino.util.inference import load_model, predict  # noqa: E402
except (ModuleNotFoundError, ImportError):
    print(
        "Could not import groundingdino. This is OK if you are only using the client."
    )
sys.path.pop(0)

GROUNDING_DINO_CONFIG = (
    "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_WEIGHTS = "data/groundingdino_swint_ogc.pth"
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        caption: str = CLASSES,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.caption = caption
        self.device = device

    @staticmethod
    def _preprocess_np(image: np.ndarray) -> torch.Tensor:
        """
        image: uint8 HWC numpy array
        return: float32 CHW torch tensor (contiguous)
        """
        # 1. First convert to float32, avoid repeated conversions later
        img = image.astype(np.float32, copy=False)
        # 2. In-place normalization
        img = (img / 255.0 - GroundingDINO._MEAN) / GroundingDINO._STD
        # 3. HWC -> CHW
        img = np.transpose(img, (2, 0, 1))  # returns view, no copy
        # 4. Ensure contiguous + zero-copy wrapping
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        return tensor

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
        verbose: bool = False,
        strict: bool = False,
    ) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Original preprocessing
        # image_tensor = F.to_tensor(image)
        # image_transformed = F.normalize(
        #     image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        image_transformed = GroundingDINO._preprocess_np(image)

        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption

        if verbose:
            print(
                "GroundingDINO is detecting. Strict:",
                strict,
                ". Caption:",
                caption_to_use,
            )

        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # Remove detections whose class names do not exactly match the provided classes
        if strict:
            classes = caption_to_use[: -len(" .")].split(" . ")
            detections.filter_by_class(classes)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(
        self,
        image_numpy: np.ndarray,
        caption: Optional[str] = "",
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
        verbose: bool = False,
        strict: bool = False,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            verbose=verbose,
            strict=strict,
        )
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


class GroundingDINOInterface:
    def __init__(
        self,
        http_mode: bool = VLM_GROUNDING_DINO_HTTP_MODE,
        port: int = VLM_GROUNDING_DINO_PORT,
    ):
        self.http_mode = http_mode
        if self.http_mode:
            self.client = GroundingDINOClient(port)
            self.model = None
        else:
            self.model = GroundingDINO()
            self.client = None

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,
        box_threshold: Optional[float] = 0.35,
        text_threshold: Optional[float] = 0.25,
        verbose: bool = False,
        strict: bool = False,
    ) -> ObjectDetections:
        if self.http_mode:
            return self.client.predict(
                image_numpy=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                verbose=verbose,
                strict=strict,
            )
        else:
            return self.model.predict(
                image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                verbose=verbose,
                strict=strict,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=VLM_GROUNDING_DINO_PORT)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(
                image,
                caption=payload["caption"],
                box_threshold=payload["box_threshold"],
                text_threshold=payload["text_threshold"],
                verbose=payload["verbose"],
                strict=payload["strict"],
            ).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
