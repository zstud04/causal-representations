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


import random

def create_binary_choice_df(
    stim_df,
    input_prompt: str,
    correct_opt_col: str,
    incorrect_opt_col: str,
    prompt_template: str,
):
    """
    Create a formatted binary-choice prompt with randomized A/B assignment per row.
    Keeps original columns unchanged and adds `formatted_prompt`.
    """
    df = stim_df.copy()

    def _format_row(row):
        if random.random() < 0.5:
            a, b = row[correct_opt_col], row[incorrect_opt_col]
        else:
            a, b = row[incorrect_opt_col], row[correct_opt_col]

        return prompt_template.format(
            prompt=row[input_prompt],
            a=a,
            b=b,
        )

    df["formatted_prompt"] = df.apply(_format_row, axis=1)
    return df




def dest_csv_append(formatted_df: pd.DataFrame, out_dest_path: str) -> None:
    """
    Append columns from formatted_df to an existing CSV at out_dest_path.
    Only adds missing columns; never overwrites existing ones.
    Creates the CSV if it does not exist.
    """
    if os.path.exists(out_dest_path):
        try:
            existing_df = pd.read_csv(out_dest_path)

            if len(existing_df) != len(formatted_df):
                raise ValueError(
                    f"Row count mismatch: existing={len(existing_df)}, new={len(formatted_df)}"
                )

            for col in formatted_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = formatted_df[col]

            existing_df.to_csv(out_dest_path, index=False)

        except Exception as e:
            raise RuntimeError(
                f"dest_csv_append failed for '{out_dest_path}': {e}"
            ) from e

    else:
        os.makedirs(os.path.dirname(out_dest_path), exist_ok=True)
        formatted_df.to_csv(out_dest_path, index=False)
