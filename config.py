import os
from pathlib import Path

import torch


def _repo_root():
    return Path(__file__).resolve().parent


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _resolve_existing_path(*candidates):
    resolved = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved.append(Path(candidate).expanduser())

    if not resolved:
        return None

    for candidate in resolved:
        if candidate.exists():
            return candidate

    return resolved[0]


class Config:
    REPO_ROOT = _repo_root()
    LEGACY_BASE_DIR = Path("/root/autodl-tmp/mllm_sam_project")

    # 基础路径
    BASE_DIR = _resolve_existing_path(
        os.environ.get("M_SAGENT_BASE_DIR"),
        REPO_ROOT,
        LEGACY_BASE_DIR,
    )
    OUTPUT_DIR = Path(os.environ.get("M_SAGENT_OUTPUT_DIR", BASE_DIR / "outputs")).expanduser()
    EXAMPLES_DIR = Path(os.environ.get("M_SAGENT_EXAMPLES_DIR", BASE_DIR / "examples")).expanduser()

    # 模型路径
    QWEN_MODEL_PATH = str(_resolve_existing_path(
        os.environ.get("M_SAGENT_QWEN_MODEL_PATH"),
        BASE_DIR / "Qwen2.5-VL-7B-Instruct",
        BASE_DIR.parent / "Qwen2.5-VL-7B-Instruct",
        Path("/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"),
    ))
    SAM3_MODEL_PATH = str(_resolve_existing_path(
        os.environ.get("M_SAGENT_SAM3_MODEL_PATH"),
        BASE_DIR / "sam3",
        Path("/root/autodl-tmp/sam3"),
    ))
    SAM3_CHECKPOINT_PATH = str(_resolve_existing_path(
        os.environ.get("M_SAGENT_SAM3_CHECKPOINT_PATH"),
        Path(SAM3_MODEL_PATH) / "sam3.pt",
        Path("/root/autodl-tmp/modelscope_cache_sam3/facebook/sam3/sam3.pt"),
    ))
    SAM3_BPE_PATH = str(_resolve_existing_path(
        os.environ.get("M_SAGENT_SAM3_BPE_PATH"),
        Path(SAM3_MODEL_PATH) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ))

    # 文本提示词路径
    SYSTEM_PROMPT = _resolve_existing_path(
        os.environ.get("M_SAGENT_SYSTEM_PROMPT"),
        BASE_DIR / "prompts" / "system_prompt_en.txt",
        BASE_DIR / "1.txt",
        BASE_DIR / "sam3" / "sam3" / "agent" / "system_prompts" / "system_prompt.txt",
    )
    SYSTEM_PROMPT_ITERATIVE_CHECKING = _resolve_existing_path(
        os.environ.get("M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING"),
        BASE_DIR / "prompts" / "system_prompt_iterative_checking_en.txt",
        BASE_DIR / "sam3" / "sam3" / "agent" / "system_prompts" / "system_prompt_iterative_checking.txt",
    )

    # 工具调用中间记录保存路径
    TOOL_CALLS_LOG = Path(os.environ.get("M_SAGENT_TOOL_CALLS_LOG", BASE_DIR / "tool_calls_log")).expanduser()

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 工具配置
    MAX_TOOL_CALLS = 5
    CONFIDENCE_THRESHOLD = 0.7

    # 网格配置
    GRID_ROWS = 5
    GRID_COLS = 5
    MIN_CELL_SIZE = 50

    # 图像分辨率配置
    MLLM_MIN_PIXELS = _env_int("M_SAGENT_MLLM_MIN_PIXELS", 3000000)
    MLLM_MAX_PIXELS = _env_int("M_SAGENT_MLLM_MAX_PIXELS", 12845056)

    # 定位配置
    MAX_POINTS = 5
    MIN_POINTS = 2

    # TrainAdapter 服务配置
    TRAIN_ADAPTER_ENABLED = _env_bool("TRAIN_ADAPTER_ENABLED", True)
    TRAIN_ADAPTER_URL = os.environ.get("TRAIN_ADAPTER_URL", "http://127.0.0.1:8765").rstrip("/")
    TRAIN_ADAPTER_TIMEOUT = _env_float("TRAIN_ADAPTER_TIMEOUT", 10.0)
    TRAIN_ADAPTER_DYNAMIC_TOPK = _env_bool("TRAIN_ADAPTER_DYNAMIC_TOPK", True)
    TRAIN_ADAPTER_ABS_THRESHOLD = _env_float("TRAIN_ADAPTER_ABS_THRESHOLD", 0.35)
    TRAIN_ADAPTER_REL_RATIO = _env_float("TRAIN_ADAPTER_REL_RATIO", 0.75)
    TRAIN_ADAPTER_MIN_K = _env_int("TRAIN_ADAPTER_MIN_K", 1)
    TRAIN_ADAPTER_MAX_K = _env_int("TRAIN_ADAPTER_MAX_K", 6)
    TRAIN_ADAPTER_MIN_SCORE_FOR_USE = _env_float("TRAIN_ADAPTER_MIN_SCORE_FOR_USE", 0.35)

    # 概念生成配置
    MIN_CONCEPTS = 3
    MAX_CONCEPTS = 5

    @classmethod
    def setup_dirs(cls):
        """创建必要目录。"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        cls.TOOL_CALLS_LOG.mkdir(parents=True, exist_ok=True)

        sam_results = Path(cls.SAM3_MODEL_PATH).expanduser() / "results"
        sam_results.mkdir(parents=True, exist_ok=True)
        
