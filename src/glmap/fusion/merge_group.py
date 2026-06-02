import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from mllm.ollama_mllm import OllamaMLLM

from ..config import GroupFusionConfig
from ..core.data_structures import GroupData, InstanceData

logger = logging.getLogger(__name__)

_ollama_mllm: Optional[OllamaMLLM] = None


def _get_mllm() -> OllamaMLLM:
    global _ollama_mllm
    if _ollama_mllm is None:
        _ollama_mllm = OllamaMLLM(model_id=GroupFusionConfig.MERGE_DESCRIPTION_MODEL_ID)
    return _ollama_mllm


def get_text_embedding(
    text: str,
    model: str = GroupFusionConfig.TEXT_EMBEDDING_MODEL_ID,
) -> float:
    return _get_mllm().embed_text(text, model)


class _UF:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        fx, fy = self.find(x), self.find(y)
        if fx != fy:
            self.parent[fy] = fx


def compute_merge_groups(groups: Sequence[GroupData]) -> List[set[int]]:
    """
    Input: Variable length GroupData sequence
    Returns: list[set[group_id]], each set contains group IDs to merge, each ID appears only once
    """
    if not groups:
        return []

    n = len(groups)
    gid2idx: Dict[str, int] = {g.group_id: i for i, g in enumerate(groups)}
    idx2gid: Dict[int, str] = {i: g.group_id for i, g in enumerate(groups)}

    # Calculate embedding vectors
    embeddings: List[np.ndarray] = []
    for group in groups:
        emb = get_text_embedding(f"Group: {group.description}")
        embeddings.append(emb)

    # 2. Construct "instance overlap" adjacency table
    instance2idx: Dict[int, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        for inst_id in group.instance_ids:
            instance2idx[inst_id].append(idx)

    uf = _UF(n)
    visited_pairs = set()
    for idx_list in instance2idx.values():
        for i in idx_list:
            for j in idx_list:
                if i == j:
                    continue
                if (i, j) in visited_pairs:
                    continue
                visited_pairs.add((i, j))
                visited_pairs.add((j, i))

                # Calculate cosine similarity (embedding already cached)
                sim = float(
                    np.dot(embeddings[i], embeddings[j])
                    / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        + 1e-12
                    )
                )
                if sim >= GroupFusionConfig.TEXT_EMBEDDING_SIMILARITY_THRESH:
                    uf.union(i, j)

    # 4. Collect results
    root2ids = defaultdict(set)
    for i in range(n):
        root2ids[uf.find(i)].add(idx2gid[i])

    return list(root2ids.values())


def generate_group_description(
    instance_datas: List[InstanceData],
) -> str:
    instance_sentences = ""
    for instance in instance_datas:
        instance_sentences += (
            f"Category: {instance.category}, Description: {instance.description}\n"
        )
    result = _get_mllm().generate_group_description(instance_sentences)
    return result["group_description"]
