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
import random

import torch
##paths/config
os.environ["TORCH_USE_CUDA_DSA"] = "1"

with open("config/models.json", "r", encoding="utf-8") as f:
    MODELS_CFG = json.load(f)
    
with open("config/instructions.json", "r", encoding="utf-8") as f:
    INSTRUCTIONS_CFG = json.load(f)
    
##local imports

#core
import core.prompt_enums as prompt_enums
from core.hf_wrapper import HFWrapper
from core.api_wrapper import APIWrapper, APIResult


#utils
import utils.model_utils as model_utils
import utils.stim_utils as stim_utils
import utils.str_utils as str_utils

random.seed(42)


def general_eval(model_str: str, stim_csv_path: str, prompt_key: str, out_dest_path: str):

    formatted_df = stim_utils.create_binary_choice_df(
        pd.read_csv(stim_csv_path),
        "prompt",
        "a",
        "b",
        INSTRUCTIONS_CFG[prompt_key]
    )

    llm, llm_cfg = model_utils.load_model(model_str)

    has_reasoning = llm_cfg.get("reasoning", False)
    is_hf = llm_cfg.get("type", None) == "hf"

    # existing metrics
    formatted_df[f"logit_a_{model_str}"] = np.nan
    formatted_df[f"logit_b_{model_str}"] = np.nan
    formatted_df[f"logit_diff_{model_str}"] = np.nan
    formatted_df[f"logit_entropy_{model_str}"] = np.nan
    formatted_df[f"correct_{model_str}"] = np.nan

    # NEW: model outputs
    formatted_df[f"output_{model_str}"] = None
    formatted_df[f"reasoning_{model_str}"] = None

    for idx, row in tqdm(formatted_df.iterrows(), total=len(formatted_df)):
        max_toks = 25000 if (has_reasoning or not is_hf) else 10
        prompt_input = row["formatted_prompt"]
        print(prompt_input)

        ans_a = row["a"]
        ans_b = row["b"]

        ans = llm.generate(prompt_input, max_toks)

        # -------------------------
        # APIResult vs raw string
        # -------------------------
        if isinstance(ans, APIResult):
            ans_text = ans.text
            ans_reasoning = ans.reasoning
            print(ans_text)
            print(ans_reasoning)
        else:
            ans_text = ans
            ans_reasoning = None

        print("\nTHE ANSWER:---")
        print(ans_text)
        print(ans_reasoning)
        print("\n----")

        formatted_df.at[idx, f"output_{model_str}"] = ans_text
        formatted_df.at[idx, f"reasoning_{model_str}"] = ans_reasoning

        if is_hf:
            logits = llm.get_logit_outs(prompt_input)
            logit_a = llm.get_logit_for_tok(logits, ans_a)
            logit_b = llm.get_logit_for_tok(logits, ans_b)

            logits_last = logits[0, -1]
            
            entropy = Categorical(logits=logits_last.float()).entropy().item()

            formatted_df.at[idx, f"logit_a_{model_str}"] = logit_a
            formatted_df.at[idx, f"logit_b_{model_str}"] = logit_b
            formatted_df.at[idx, f"logit_diff_{model_str}"] = logit_a - logit_b
            formatted_df.at[idx, f"logit_entropy_{model_str}"] = entropy

        is_correct = str_utils.is_correct_match(ans_text, ans_a, ans_b)
        formatted_df.at[idx, f"correct_{model_str}"] = is_correct

    stim_utils.dest_csv_append(formatted_df, out_dest_path)



    

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