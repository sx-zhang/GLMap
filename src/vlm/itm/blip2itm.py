from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from vlm.config import VLM_BLIP2ITM_HTTP_MODE, VLM_BLIP2ITM_PORT

from ..server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")


class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    _SIZE = 224
    _MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.model.eval()
        self.device = device

    @staticmethod
    def _preprocess_np(image: np.ndarray) -> torch.Tensor:
        """
        image: uint8 HWC numpy array
        return: float32 CHW torch tensor, values consistent with BlipImageEvalProcessor
        """
        # 1. First resize to 224×224, **bicubic + antialias**
        #    OpenCV's INTER_CUBIC is equivalent to PIL's bicubic (default antialias=True)
        img = cv2.resize(
            image, (BLIP2ITM._SIZE, BLIP2ITM._SIZE), interpolation=cv2.INTER_CUBIC
        )

        # 2. Convert to float32 + normalize
        img = img.astype(np.float32, copy=False) / 255.0
        img = (img - BLIP2ITM._MEAN) / BLIP2ITM._STD

        # 3. HWC → CHW + ensure contiguous
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(img))

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        # In the original preprocessing, self.vis_processors["eval"] is `lavis.processors.blip_processors.BlipImageEvalProcessor`
        # [Resize(size=(224, 224), interpolation=bicubic, max_size=None, antialias=True), ToTensor(), Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
        # pil_img = Image.fromarray(image)
        # img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        img = BLIP2ITM._preprocess_np(image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            cosine = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()
        return cosine

    def itm_scores(self, image: np.ndarray, txt: str) -> np.ndarray:
        # pil_img = Image.fromarray(image)
        # img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        img = BLIP2ITM._preprocess_np(image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

        itm_score = itm_scores[:, 1].item()
        return itm_score


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> np.ndarray:
        print(f"Question of blip2 is:{txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["itm score"])


class BLIP2ITMInterface:
    def __init__(self, http_mode=VLM_BLIP2ITM_HTTP_MODE, port: int = VLM_BLIP2ITM_PORT):
        self.http_mode = http_mode
        if self.http_mode:
            self.client = BLIP2ITMClient(port)
            self.model = None
        else:
            self.model = BLIP2ITM()
            self.client = None

    def cosine(self, image: np.ndarray, txt: str) -> float:
        if self.http_mode:
            return self.client.cosine(image, txt)
        else:
            return self.model.cosine(image, txt)

    def itm_score(self, image: np.ndarray, txt: str) -> np.ndarray:
        if self.http_mode:
            return self.client.itm_score(image, txt)
        else:
            return self.model.itm_scores(image, txt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=VLM_BLIP2ITM_PORT)
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {
                "response": self.cosine(image, payload["txt"]),
                "itm score": self.itm_scores(image, payload["txt"]),
            }

    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
