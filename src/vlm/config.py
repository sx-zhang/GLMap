import os
import warnings

# Ignore FutureWarnings from Transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Ignore UserWarnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Ignore FutureWarnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Ignore specific torch.cuda.amp.autocast deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)


# VLM service port environment variable configuration
VLM_GROUNDING_DINO_PORT = int(os.environ.get("VLM_GROUNDING_DINO_PORT", 12181))
VLM_BLIP2ITM_PORT = int(os.environ.get("VLM_BLIP2ITM_PORT", 12182))
VLM_MOBILE_SAM_PORT = int(os.environ.get("VLM_MOBILE_SAM_PORT", 12183))
VLM_YOLOV7_PORT = int(os.environ.get("VLM_YOLOV7_PORT", 12184))

# VLM operation mode environment variable configuration (True indicates HTTP mode, False indicates local mode)
DEFAULT_HTTP_MODE = "true"
VLM_GROUNDING_DINO_HTTP_MODE = (
    os.environ.get("VLM_GROUNDING_DINO_MODE", DEFAULT_HTTP_MODE).lower() == "true"
)
VLM_MOBILE_SAM_HTTP_MODE = (
    os.environ.get("VLM_MOBILE_SAM_HTTP_MODE", DEFAULT_HTTP_MODE).lower() == "true"
)
VLM_BLIP2ITM_HTTP_MODE = (
    os.environ.get("VLM_BLIP2ITM_HTTP_MODE", DEFAULT_HTTP_MODE).lower() == "true"
)
VLM_YOLOV7_HTTP_MODE = (
    os.environ.get("VLM_YOLOV7_HTTP_MODE", DEFAULT_HTTP_MODE).lower() == "true"
)
