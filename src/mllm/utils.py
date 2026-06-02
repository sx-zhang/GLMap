import base64
import logging
from io import BytesIO
from typing import Optional

from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> Optional[str]:
    if image is None:
        return None

    try:
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format=format)
            image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            return base64_image
        else:
            logging.warning(f"Unsupported image format: {type(image)}")
            return None

    except Exception as e:
        logging.error(f"Image processing failed: {e}")
        return None


def image_to_base64_data_uri(image: Image.Image, format: str = "PNG") -> Optional[str]:
    base64_image = image_to_base64(image, format)
    if base64_image:
        return f"data:image/{format.lower()};base64,{base64_image}"
    return None
