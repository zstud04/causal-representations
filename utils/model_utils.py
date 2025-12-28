from torch import Tensor, nn

from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import gc
import requests
import os
import sys
import json
import inspect

import torch
##paths/config
os.environ["TORCH_USE_CUDA_DSA"] = "1"

with open("config/models.json", "r", encoding="utf-8") as f:
    MODELS_CFG = json.load(f)
    
##local imports
import core.prompt_enums as prompt_enums
from core.hf_wrapper import HFWrapper
from core.api_wrapper import APIWrapper 


def mem_cleanup(llm):

    if llm is None:
        return

    if hasattr(llm, "model"):
        try:
            llm.model.reset_hooks()
        except Exception:
            pass
        del llm.model

    if hasattr(llm, "tokenizer"):
        del llm.tokenizer

    del llm
    gc.collect()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def load_model(model_str):
    """
    Load model from json (either HF or API wrapper). Return wrapper, json obj
    """
    model_cfg = MODELS_CFG.get(model_str)
    if model_cfg is None:
        raise KeyError(f"Model not found in config: {model_str}")
    print(model_cfg)

    if model_cfg["type"] == "hf":
        llm = HFWrapper(
            model_cfg["name"],
            model_cfg["path"],
            model_cfg["tl_support"],
            cache_dir=os.getenv("CACHE_DIR")
        )

    else:
        key = {
            "openai": os.getenv("OPENAI_KEY"),
            "gemini": os.getenv("GEMINI_KEY"),
            "deepseek": os.getenv("DEEPSEEK_KEY"),   # ← ADD THIS
        }.get(model_cfg["type"])

        if key is None:
            raise ValueError(
                f"Missing API key for provider '{model_cfg['type']}'"
            )

        llm = APIWrapper(
            model_cfg["type"],
            model_cfg["path"],
            key,
        )

    return llm, model_cfg

