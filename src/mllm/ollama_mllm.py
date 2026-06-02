import logging
from typing import List, Optional, Union

import ollama
from PIL import Image

from .mllm_base import MLLM_Mixin
from .utils import image_to_base64


class OllamaMLLM(MLLM_Mixin):
    def __init__(
        self,
        model_id: str = "gemma3:27b",
    ):
        self.model_id = model_id
        self.logger = logging.getLogger(f"{__class__.__name__}-{model_id}")

        local_models = {m["model"] for m in ollama.list()["models"]}
        if model_id not in local_models:
            self.logger.warning(
                f"Model {model_id} not found locally, please run `ollama pull {model_id}` first"
            )
        else:
            self.logger.info(f"Model {model_id} ready in ollama")

    def embed_text(self, text: str, model_id: str = None) -> List[float]:
        model_id = model_id or self.model_id
        try:
            response = ollama.embeddings(
                model=model_id,
                prompt=text,
            )
            return response.get("embedding", [])
        except Exception as e:
            self.logger.error(f"Ollama {model_id} text embedding failed: {str(e)}")
            return []

    def chat(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        messages = []

        if image is not None:
            base64_image = image_to_base64(image)
            if base64_image:
                # Build message containing image
                image_message = {
                    "role": "user",
                    "content": text_prompt,
                    "images": [base64_image],
                }
                messages.append(image_message)
            else:
                self.logger.warning(
                    "Image processing failed, falling back to text-only input"
                )
                messages.append({"role": "user", "content": text_prompt})
        else:
            messages.append({"role": "user", "content": text_prompt})

        try:
            response = ollama.chat(model=self.model_id, messages=messages)
            answer = response["message"]["content"]
            return answer
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            if image is not None:
                self.logger.warning("Attempting to fall back to text-only mode")
                messages = [{"role": "user", "content": text_prompt}]
                response = ollama.chat(model=self.model_id, messages=messages)
                answer = response["message"]["content"]
                return answer
            raise

    def chat_images(
        self,
        content: List[Union[str, Image.Image]],
    ) -> str:
        """
        Support text-image alternating input conversation method
        Args:
            content: Alternating text and image list, e.g., [text1, image1, text2, image2, ...]
        """
        if not content:
            return ""

        # Process content, convert text and images into multiple independent messages
        messages = []
        has_image = False
        current_text = None
        current_images = []

        for item in content:
            if isinstance(item, str):
                if current_text is not None:
                    if current_images:
                        messages.append(
                            {
                                "role": "user",
                                "content": current_text,
                                "images": current_images,
                            }
                        )
                        has_image = True
                    else:
                        messages.append({"role": "user", "content": current_text})
                    current_images = []
                current_text = item
            elif isinstance(item, Image.Image):
                base64_img = image_to_base64(item)
                if base64_img:
                    current_images.append(base64_img)
                    has_image = True
                else:
                    self.logger.warning("Image processing failed, skipping this image")
            else:
                self.logger.warning(f"Unsupported content type: {type(item)}")

        # Process the last accumulated text and images
        if current_text is not None:
            if current_images:
                messages.append(
                    {"role": "user", "content": current_text, "images": current_images}
                )
                has_image = True
            else:
                messages.append({"role": "user", "content": current_text})

        if not has_image:
            self.logger.warning("No valid images, falling back to text-only input")
            # Merge all texts
            text_content = " ".join([item for item in content if isinstance(item, str)])
            return self.chat(text_prompt=text_content, image=None)

        try:
            response = ollama.chat(model=self.model_id, messages=messages)
            answer = response["message"]["content"]
            return answer
        except Exception as e:
            self.logger.error(f"Multi-image API call failed: {e}")
            self.logger.warning("Attempting to fall back to text-only mode")
            text_content = " ".join([item for item in content if isinstance(item, str)])
            return self.chat(text_prompt=text_content, image=None)
