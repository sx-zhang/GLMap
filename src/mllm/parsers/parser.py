import ast
import re
from typing import Any, Dict, List

from .filter import exclude_objects_by_category

EXCLUDE_CATEGORIES = [
    "background",
    "wall",
    "door",
    "window",
    "mirror",
    "outlet",
    "baseboard",
    "carpet",
    "curtains",
]


class ParseError(RuntimeError):
    """Exception for post-processing stage, for upstream catching"""

    pass


def _extract_dict(raw: str) -> Dict[str, Any]:
    """Extract outermost dictionary from model return and literal_eval"""
    if not raw:
        raise ParseError("Model return is empty")

    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("Could not locate dictionary structure")

    try:
        data = ast.literal_eval(m.group(0))
    except Exception as e:
        raise ParseError(f"ast.literal_eval failed: {e}")

    if not isinstance(data, dict):
        raise ParseError("Top-level structure is not dict")
    return data


def _check_field(data: Dict[str, Any], key: str, expected_type: type) -> Any:
    """Generic field type checking"""
    val = data.get(key)
    if val is None:
        raise ParseError(f'Missing field "{key}"')
    if not isinstance(val, expected_type):
        raise ParseError(
            f'Field "{key}" should be {expected_type.__name__}, actual type {type(val)}'
        )
    return val


def parse_merge_instance_output(raw: str) -> Dict[str, Any]:
    """
    Parse merge instance output, extract and validate merged_type and merged_description fields
    {{
    "merged_type": "merged type",
    "merged_description": "merged description"
    }}
    """
    if not raw:
        raise ParseError("Model return is empty")

    # Extract outermost dictionary
    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("Could not locate dictionary structure")
    dict_str = m.group(0)

    # Safely parse to Python dictionary
    try:
        data = ast.literal_eval(dict_str)
    except Exception as e:
        raise ParseError(f"ast.literal_eval failed: {e}")

    if not isinstance(data, dict):
        raise ParseError("Top-level structure is not dict")

    # Field validation and type conversion
    def _check_str(key: str) -> str:
        val = data.get(key)
        if val is None:
            raise ParseError(f'Missing field "{key}"')
        if not isinstance(val, str):
            raise ParseError(f'Field "{key}" should be str, actual type {type(val)}')
        return val

    cleaned = {
        "merged_category": _check_str("merged_category"),
        "merged_description": _check_str("merged_description"),
    }

    return cleaned


def parse_instance_similarity_output(raw: str) -> Dict[str, Any]:
    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        raise ParseError(f"ast.literal_eval failed: {e}")

    if not isinstance(data, dict):
        raise ParseError("Top-level structure is not dict")

    # Field validation and type conversion
    def _check_str(key: str) -> str:
        val = data.get(key)
        if not isinstance(val, str):
            raise ParseError(f'Field "{key}" should be str, actual type {type(val)}')
        return val

    def _check_bool(key: str) -> bool:
        val = data.get(key)
        if not isinstance(val, bool):
            raise ParseError(f'Field "{key}" should be bool, actual type {type(val)}')
        return val

    cleaned = {
        "reasoning": _check_str("reasoning"),
        "should_merge": _check_bool("should_merge"),
    }

    return cleaned


def parse_grounding_output(raw: str) -> Dict[str, Any]:
    """
    Parse string returned by model under FOR_GROUNDING_ZH prompt, return structured dict.
    Top-level fields refer to FOR_GROUNDING prompt in prompts
    If parsing fails, raise ParseError.
    """
    if not raw:
        raise ParseError("Model return is empty")

    # Extract outermost dictionary, non-greedy match between first { and last }
    m = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    if not m:
        raise ParseError("Could not locate dictionary structure")
    dict_str = m.group(0)

    # Safely parse to Python dict
    try:
        data = ast.literal_eval(dict_str)
    except Exception as e:
        raise ParseError(f"ast.literal_eval failed: {e}")

    if not isinstance(data, dict):
        raise ParseError("Top-level structure is not dict")

    # Field validation and type conversion
    def _check_str(key: str) -> str:
        val = data.get(key)
        if not isinstance(val, str):
            raise ParseError(f'Field "{key}" should be str, actual type {type(val)}')
        return val

    def _check_list_dict(key: str) -> List[Dict[str, Any]]:
        val = data.get(key)
        if not isinstance(val, list):
            raise ParseError(f'Field "{key}" should be list, actual type {type(val)}')
        for idx, item in enumerate(val):
            if not isinstance(item, dict):
                raise ParseError(f'Field "{key}" item {idx} is not dict')
        return val

    cleaned = {
        "scene_summary": _check_str("scene_summary"),
        "objects": _check_list_dict("objects"),
        "groups": _check_list_dict("groups"),
    }

    cleaned = exclude_objects_by_category(
        cleaned=cleaned, exclude_category=EXCLUDE_CATEGORIES
    )

    return cleaned


def get_objects_type_list(objects: List[Dict[str, Any]]) -> List[str]:
    if not objects:
        raise ParseError("Input is empty")
    types = [obj["type"] for obj in objects]
    return types


def parse_renew_instance_output(raw: str) -> Dict[str, Any]:
    """
    Parse renew_instance output
    Returns: {"category": str, "description": str}
    """
    data = _extract_dict(raw)
    return {
        "category": _check_field(data, "category", str),
        "description": _check_field(data, "description", str),
    }


def parse_group_description_output(raw: str) -> Dict[str, Any]:
    """
    Parse generate_group_description output
    Returns: {"group_description": str}
    """
    data = _extract_dict(raw)
    return {
        "group_description": _check_field(data, "group_description", str),
    }


def parse_value_output(raw: str) -> Dict[str, Any]:
    """
    Parse instance/group target value output
    Returns: {"answer": float}
    """
    data = _extract_dict(raw)
    return {
        "answer": _check_field(data, "answer", float),
    }
