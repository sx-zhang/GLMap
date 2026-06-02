import logging
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.spatial import cKDTree

from mllm.ollama_mllm import OllamaMLLM

from ..config import InstanceFusionConfig
from ..core.data_structures import InstanceData

logger = logging.getLogger(__name__)

_ollama_mllm: Optional[OllamaMLLM] = None


def _get_mllm() -> OllamaMLLM:
    global _ollama_mllm
    if _ollama_mllm is None:
        _ollama_mllm = OllamaMLLM(
            model_id=InstanceFusionConfig.MERGE_RESULT_LLM_MODEL_ID
        )
    return _ollama_mllm


def get_text_embedding_similarity(
    text_1: str,
    text_2: str,
    model: str = InstanceFusionConfig.TEXT_EMBEDDING_MODEL_ID,
) -> float:
    """
    Calculate text similarity using embedding vectors from Ollama library
    """
    emb1 = _get_mllm().embed_text(text_1, model)
    emb2 = _get_mllm().embed_text(text_2, model)

    if not emb1 or not emb2:
        raise ValueError(
            f"Embedding failed, cannot calculate similarity for {text_1} and {text_2}"
        )

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    logger.debug(
        "Similarity between text %s and %s is %.3f",
        text_1,
        text_2,
        similarity,
    )
    return similarity


def check_point_cloud_connectivity(
    point_cloud1: np.ndarray,
    point_cloud2: np.ndarray,
    dist_thresh: float = InstanceFusionConfig.CLOUD_POINT_CONNECTITY_THRESH,
    count_thresh: int = InstanceFusionConfig.CLOUD_POINT_CONNECTITY_COUNT_THRESH,
) -> bool:
    if point_cloud1.size == 0 or point_cloud2.size == 0:
        return False

    # Build KD-Tree
    tree = cKDTree(point_cloud2)

    # Batch search radius neighborhood
    # idx_list returned is a list of length N, each element is an array storing neighborhood indices of corresponding point
    idx_list = tree.query_ball_point(point_cloud1, r=dist_thresh)

    # Count number of points with neighbors
    match_count = sum(len(neighbors) > 0 for neighbors in idx_list)

    return match_count >= count_thresh


def check_instance_should_merge(
    default_instance: InstanceData,
    new_instance: InstanceData,
) -> bool:
    """Check if instances should be merged"""
    connectivity = check_point_cloud_connectivity(
        default_instance.point_cloud, new_instance.point_cloud
    )
    if not connectivity:
        return False

    if default_instance.category.lower() == new_instance.category.lower():
        return True

    query_text_1 = f"category: {default_instance.category}, description: {default_instance.description}"
    query_text_2 = (
        f"category: {new_instance.category}, description: {new_instance.description}"
    )

    similarity = get_text_embedding_similarity(query_text_1, query_text_2)
    return similarity >= InstanceFusionConfig.TEXT_EMBEDDING_SIMILARITY_THRESH


def renew_instance_description(
    history_categories_descriptions: str,
) -> None:
    """Update instance category and description"""
    result = _get_mllm().renew_instance_description(
        history_categories_descriptions=history_categories_descriptions,
    )
    if result and "category" in result and "description" in result:
        return result

    logger.error("Failed to renew instance type and description")
    return {}


def merge_instance_descriptions(
    category_1: str,
    description_1: str,
    category_2: str,
    description_2: str,
) -> Dict[str, str]:
    """Merge two similar instances"""
    result = _get_mllm().merge_instance_descriptions(
        category_1=category_1,
        description_1=description_1,
        category_2=category_2,
        description_2=description_2,
    )
    if result and "merged_category" in result and "merged_description" in result:
        return result

    logger.error("Failed to merge instances %s and %s", category_1, category_2)
    return {}
