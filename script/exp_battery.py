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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

with open("config/models.json", "r", encoding="utf-8") as f:
    MODELS_CFG = json.load(f)
    
##local imports

#core
import core.prompt_enums as prompt_enums
from core.hf_wrapper import HFWrapper
from core.api_wrapper import APIWrapper 


#utils
import utils.model_utils as model_utils

def general_eval(model_path: str):
    """
    Run general eval for full model suite
    """
    with open(model_path, "r", encoding="utf-8") as f:
        model_list = [line.strip() for line in f if line.strip()]

    for model_str in model_list:
        llm = model_utils.load_model(model_str)
        print(llm)
        
    model_utils.mem_cleanup()  

    

def wm_eval(model_path:str):
    """
    Run world model eval for full model suite
    """
    pass

def general_explanations():
    """
    Get explanations for large model suite
    """
    pass

def explanation_code():
    """
    classify model + human explanations based on pre-specified code scheme
    """
    pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script/exp_battery.py <function_name> [args...]")
        sys.exit(1)

    fn_name = sys.argv[1]
    fn_args = sys.argv[2:]

    if fn_name not in globals() or not callable(globals()[fn_name]):
        raise ValueError(f"Unknown function: {fn_name}")

    globals()[fn_name](*fn_args)