from typing import Any, Callable, Dict, List, Optional, Union, cast

import json_repair
import numpy as np
from PIL import Image

from .parsers import (
    parse_grounding_output,
    parse_group_description_output,
    parse_instance_similarity_output,
    parse_merge_instance_output,
    parse_renew_instance_output,
    parse_value_output,
)
from .prompts import (
    FOR_GENERATE_GROUP_DESCRIPTION,
    FOR_GROUNDING,
    FOR_GROUP_TARGET_VALUE,
    FOR_INSTANCE_TARGET_VALUE,
    FOR_MERGE_INSTANCE_CATEGORY_DESCRIPTION,
    FOR_RENEW_INSTANCE_CATEGORY_DESCRIPTION,
    FOR_SAME_MAP_COORD_INSTANCE_SIMILARITY,
)


def default_parser(raw_ansewer: str) -> Dict[str, Any]:
    json_answer = json_repair.loads(raw_ansewer)
    return json_answer  # type: ignore


class MLLM_Protocol:
    def chat(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        raise NotImplementedError("MLLM_Protocol.chat")

    def chat_images(
        self,
        content: List[Union[str, Image.Image]],
    ) -> str:
        """
        Support text-image alternating input conversation method
        Args:
            content: Alternating text and image list, e.g., [text1, image1, text2, image2, ...]
        """
        raise NotImplementedError("MLLM_Protocol.chat_images")


class MLLM_Mixin(MLLM_Protocol):
    def chat_with_postprocessing(
        self,
        text_prompt: str,
        image: Optional[Image.Image] = None,
        postprocessing_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        max_retry_times: int = 3,
    ) -> Dict[str, Any]:
        # If caller doesn't provide postprocessing function, use identity function
        if postprocessing_fn is None:
            postprocessing_fn = lambda ans: cast(Dict[str, Any], ans)

        for t in range(max_retry_times):
            answer = self.chat(text_prompt=text_prompt, image=image)
            try:
                return postprocessing_fn(answer)
            except Exception as exc:
                if hasattr(self, "logger"):
                    self.logger.error(
                        "Postprocessing failed at attempt %d: %s", t + 1, exc
                    )
                continue
        if hasattr(self, "logger"):
            self.logger.error(
                "%d retry attempts failed, returning empty dict", max_retry_times
            )
        return {}

    def grounding(self, image: Image.Image, downsample_step: int = 2) -> Dict[str, Any]:
        if downsample_step:
            image_np = np.array(image)
            image_np = image_np[::downsample_step, ::downsample_step, :]
            image = Image.fromarray(image_np)

        return self.chat_with_postprocessing(
            text_prompt=FOR_GROUNDING,
            image=image,
            postprocessing_fn=parse_grounding_output,
        )

    def compare_instance_similarity(
        self,
        instance1_type: str,
        instance1_description: str,
        instance2_type: str,
        instance2_description: str,
    ) -> Dict[str, Any]:
        text_prompt = FOR_SAME_MAP_COORD_INSTANCE_SIMILARITY.format(
            instance1_type=instance1_type,
            instance1_description=instance1_description,
            instance2_type=instance2_type,
            instance2_description=instance2_description,
        )
        return self.chat_with_postprocessing(
            text_prompt=text_prompt,
            postprocessing_fn=parse_instance_similarity_output,
        )

    def merge_instance_descriptions(
        self,
        category_1: str,
        description_1: str,
        category_2: str,
        description_2: str,
    ) -> Dict[str, str]:
        return self.chat_with_postprocessing(
            text_prompt=FOR_MERGE_INSTANCE_CATEGORY_DESCRIPTION.format(
                category_1=category_1,
                description_1=description_1,
                category_2=category_2,
                description_2=description_2,
            ),
            postprocessing_fn=parse_merge_instance_output,
        )

    def renew_instance_description(
        self,
        history_categories_descriptions: str,
    ) -> Dict[str, str]:
        return self.chat_with_postprocessing(
            text_prompt=FOR_RENEW_INSTANCE_CATEGORY_DESCRIPTION.format(
                history_categories_descriptions=history_categories_descriptions,
            ),
            postprocessing_fn=parse_renew_instance_output,
        )

    def generate_group_description(
        self,
        instance_sentences: str,
    ) -> Dict[str, Any]:
        return self.chat_with_postprocessing(
            text_prompt=FOR_GENERATE_GROUP_DESCRIPTION.format(
                instance_sentences=instance_sentences,
            ),
            postprocessing_fn=parse_group_description_output,
        )

    def instance_target_value(
        self,
        target: str,
        instance_sentence: str,
    ) -> Dict[str, Any]:
        return self.chat_with_postprocessing(
            text_prompt=FOR_INSTANCE_TARGET_VALUE.format(
                target=target,
                instance_sentence=instance_sentence,
            ),
            postprocessing_fn=parse_value_output,
        )

    def group_target_value(
        self,
        target: str,
        group_description: str,
    ) -> Dict[str, Any]:
        return self.chat_with_postprocessing(
            text_prompt=FOR_GROUP_TARGET_VALUE.format(
                target=target,
                group_description=group_description,
            ),
            postprocessing_fn=parse_value_output,
        )
