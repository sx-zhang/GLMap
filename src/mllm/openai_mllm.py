import logging
import os
from typing import List, Optional

from PIL import Image

from .mllm_base import MLLM_Mixin
from .utils import image_to_base64_data_uri


class OpenAIMLLM(MLLM_Mixin):
    def __init__(
        self,
        model_id: str = "qwen3-vl-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ):
        self.model_id = model_id
        self.base_url = base_url

        # Initialize logger
        self.logger = logging.getLogger(f"OpenAIMLLM-{model_id}")

        # Delay import openai to avoid hard dependency
        try:
            from openai import OpenAI

            # Initialize client
            if api_key is None:
                # Try to read from environment variables
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

            if api_key:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                self.logger.info(
                    f"OpenAIMLLM initialized with model: {model_id}, base_url: {base_url}"
                )
            else:
                self.logger.warning(
                    "API key not provided. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY "
                    "environment variable or pass api_key parameter. Using mock mode."
                )
                self.client = None

        except ImportError:
            self.logger.error(
                "openai package not installed. Please install with: pip install openai"
            )
            self.client = None

    def embed_text(self, text: str, model_id: str = "text-embedding-v3") -> List[float]:
        """
        Embed text (if model supports)

        Args:
            text: Text to embed
            model_id: Optional model ID

        Returns:
            Embedding vector list, returns empty list if not supported
        """
        if self.client is None:
            self.logger.error("OpenAI client not initialized")
            return []

        try:
            # Use default embedding model or specified model
            embed_model = model_id
            response = self.client.embeddings.create(input=text, model=embed_model)
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            self.logger.error(f"Text embedding failed: {e}")
            return []

    def chat(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """
        Call OpenAI compatible multimodal model for conversation

        Args:
            text_prompt: Text prompt
            image: Optional image object

        Returns:
            Model response text
        """
        if self.client is None:
            self.logger.error("OpenAI client not initialized, returning empty response")
            return ""

        # Build messages
        messages = []

        # Process image input
        image_content = None
        if image is not None:
            image_data = image_to_base64_data_uri(image)
            if image_data:
                # Build message containing image (OpenAI format)
                image_content = {"type": "image_url", "image_url": {"url": image_data}}
            else:
                self.logger.warning(
                    "Image processing failed, falling back to text-only input"
                )

        # Build user message
        user_message = {"role": "user", "content": []}

        # Add text content
        user_message["content"].append({"type": "text", "text": text_prompt})

        # Add image content (if any)
        if image_content:
            user_message["content"].append(image_content)

        messages.append(user_message)

        try:
            # Call OpenAI compatible interface
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )

            # Extract response content
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            self.logger.error(f"API call failed: {e}")

            # If image processing error, try to fall back to text-only mode
            if image is not None:
                self.logger.warning("Attempting to fall back to text-only mode")
                user_message = {"role": "user", "content": text_prompt}
                messages = [user_message]

                try:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                    )
                    answer = response.choices[0].message.content
                    return answer
                except Exception as fallback_error:
                    self.logger.error(f"Fallback mode also failed: {fallback_error}")

            raise

    def chat_images(
        self,
        content: list,
    ) -> str:
        """
        Support text-image alternating input conversation method

        Args:
            content: Alternating text and image list, e.g., [text1, image1, text2, image2, ...]
        """
        if self.client is None:
            self.logger.error("OpenAI client not initialized, returning empty string")
            return ""

        # Build messages
        messages = []

        # Build user message
        user_message = {"role": "user", "content": []}

        # Process text and images in content
        has_image = False
        for item in content:
            if isinstance(item, str):
                # Text content
                user_message["content"].append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                # Image content
                image_data = image_to_base64_data_uri(item)
                if image_data:
                    user_message["content"].append(
                        {"type": "image_url", "image_url": {"url": image_data}}
                    )
                    has_image = True
                else:
                    self.logger.warning("Image processing failed, skipping this image")
            else:
                self.logger.warning(f"Unsupported content type: {type(item)}")

        messages.append(user_message)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
            )

            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            self.logger.error(f"Multi-image API call failed: {e}")
            if has_image:
                self.logger.warning("Attempting to fall back to text-only mode")
                # Merge all texts
                text_content = " ".join(
                    [item for item in content if isinstance(item, str)]
                )
                return self.chat(text_prompt=text_content, image=None)
            raise

    def __repr__(self) -> str:
        return f"OpenAIMLLM(model_id='{self.model_id}', base_url='{self.base_url}', api_configured={self.client is not None})"
