from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from jaxtyping import Float, Int

from torch import Tensor, nn
from transformer_lens import ActivationCache, HookedTransformer, utils
import einops
import os
import sys
import math
import tqdm

import transformer_lens.utils as utils
import json
import random
import gc
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import torch
import transformers
from datasets import load_dataset
import csv
from huggingface_hub import hf_hub_download
from typing import List, Union, Optional, Callable
from IPython.display import HTML, IFrame, clear_output, display
from jaxtyping import Float, Int
from IPython.display import IFrame
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from tqdm import tqdm
import chat_templates as chat_templates


class HFWrapper:
    
    
    def __init__(
        self,
        name:str,
        path:str,
        tl_support:bool,
        device :str = "cuda",
        quantization: Optional[str] = None,
        cache_dir=None,
        enable_templates=True#whether to format with model-specific chat templates
    ):
        self.name = name
        self.path = path
        self.tl_support = tl_support
        self.device = device
        self.quantization = quantization
        self.cache_dir = cache_dir
        
        self.enable_templates=enable_templates
        #load model and tokenizer obj(none if transformerlens)
        self.model, self.tokenizer = self.__load_model(self.path, self.tl_support)

        self.prompt_template = self.__get_prompt_template(self.path)
    
    
    
    
    def __get_prompt_template(self, path: str):
        """
        Get prompt template based on model family
        """
        path_l = path.lower()
        if "gemma" in path_l:
            return chat_templates.GEMMA_TEMPLATE
        elif "llama" in path_l:
            return chat_templates.LLAMA_TEMPLATE
        else:
            return None
            
            
    def __format_and_tokenize_prompt(self, instruction: str):
        """
        Format prompt using model-specific chat template and tokenize.
        Works for both TransformerLens and base HuggingFace.
        """
        # --- format ---
        if self.prompt_template is not None and self.enable_templates:
            prompt = self.prompt_template.format(instruction=instruction)
        else:
            prompt = instruction

        # --- tokenize ---
        if self.tl_support:
            # TransformerLens: tokens only
            toks = self.model.to_tokens(prompt).to(self.device)
            return prompt, toks, None
        else:
            # HuggingFace: input_ids + attention_mask
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            )
            input_ids = enc["input_ids"].to(self.model.device)

            return input_ids

            
        
    def __load_model(self, path:str, tl_support:bool):
        """
        return transformerlens or transformer obj depending on model support, as well as tokenizer if applicable
        """
        
        if tl_support == False:
            return AutoModelForCausalLM.from_pretrained(
                path,
                cache_dir=self.cache_dir,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto"
            ), AutoTokenizer.from_pretrained(
                path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        
        else:
            return HookedTransformer.from_pretrained(
                path,
                device=self.device,
                cache_dir=self.cache_dir,
                dtype=torch.bfloat16
            ), None
            
            
    
    def get_logit_outs(self, prompt):
        """
        Get logit out distribution for prompt
        """
        if self.tl_support:
            input_ids = self.model.to_tokens(prompt).to(self.device)
            logits = self.model(input_ids, return_type="logits")
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.model.device)
            logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
        return logits
    
    
    
    def do_model_generate(self, prompt, n_toks=10):
        """
        Generate for n tokens
        """
        if self.tl_support:
            toks = self.model.to_tokens(prompt).to(self.device)
            out_toks = self.model.generate(
                toks,
                max_new_tokens=n_toks,
                do_sample=False,
                prepend_bos=False,
            )
            out_str = self.model.to_string(out_toks[0])
            return out_str
        else:
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.model.device)
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=n_toks,
                do_sample=False,
            )
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    
    
    def get_logit_lens(self, prompt, tok_pos, tok_neg):
        """
        Get logit lens umembed projection
        """
        if self.tl_support:
            if isinstance(tok_pos, str):
                tok_pos = self.model.to_single_token(tok_pos)
            if isinstance(tok_neg, str):
                tok_neg = self.model.to_single_token(tok_neg)

            toks = self.model.to_tokens(prompt).to(self.device)
            _, cache = self.model.run_with_cache(toks)
            resid = cache.stack_activation("resid_post")  # (layer, batch, pos, d_model)
            resid = resid[:, 0]                           # (layer, pos, d_model)

            direction = self.model.W_U[:, tok_pos] - self.model.W_U[:, tok_neg]  # (d_model,)
            proj = (resid @ direction)                                           # (layer, pos)
            return proj[:, -1].detach().cpu().numpy()
        else:
            if isinstance(tok_pos, str) or isinstance(tok_neg, str):
                raise ValueError("For HF mode, tok_pos/tok_neg must be token IDs (ints).")

            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.model.device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            h = out.hidden_states[1:]  # drop embedding; list length = n_layers, each (batch, seq, hidden)

            W_U = self.model.get_output_embeddings().weight  # (vocab, hidden)
            direction = (W_U[tok_pos] - W_U[tok_neg]).to(h[0].device)  # (hidden,)

            vals = []
            for layer_h in h:
                resid_last = layer_h[0, -1, :]           # (hidden,)
                vals.append((resid_last @ direction).item())
            return np.array(vals)
        
    def get_logit_for_tok(self, logits, s: str, pos: int = -1):
        tok_id = (self.tokenizer or self.model.tokenizer).encode(s, add_special_tokens=False)[0]
        return logits[0, pos, tok_id].item()
    
    def get_top_next_token_str(self, logits):
        tok_id = logits[0, -1].argmax().item()
        return (self.tokenizer or self.model.tokenizer).decode([tok_id])





            
        