from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import Optional
from transformer_lens import HookedTransformer
import core.chat_templates as chat_templates


class HFWrapper:

    def __init__(
        self,
        name: str,
        path: str,
        tl_support: bool,
        device: str = "cuda",
        quantization: Optional[str] = None,
        cache_dir=None,
        enable_templates: bool = True,
    ):
        self.name = name
        self.path = path
        self.tl_support = tl_support
        self.device = device
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.enable_templates = enable_templates

        self.model, self.tokenizer = self.__load_model(self.path, self.tl_support)

        if self.tokenizer is not None:
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.prompt_template = self.__get_prompt_template(self.path)


    def __get_prompt_template(self, path: str):
        path_l = path.lower()
        if "it" in path_l or "chat" in path_l:#find better way to do this
            if "gemma" in path_l:
                return chat_templates.GEMMA_TEMPLATE
            elif "llama" in path_l:
                return chat_templates.LLAMA_TEMPLATE
            else:
                
                return chat_templates.BASE_TEMPLATE


    def __format_prompt(self, instruction: str):
        if self.prompt_template is not None and self.enable_templates:
            return self.prompt_template.format(instruction=instruction)
        return instruction


    def __load_model(self, path: str, tl_support: bool):
        if not tl_support:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                cache_dir=self.cache_dir,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            return model, tokenizer
        else:
            return (
                HookedTransformer.from_pretrained(
                    path,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    dtype=torch.bfloat16,
                ),
                None,
            )


    def _encode(self, prompt: str):
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return {
            "input_ids": enc["input_ids"].to(self.model.device),
            "attention_mask": enc["attention_mask"].to(self.model.device),
        }


    def get_logit_outs(self, instruction: str):
        prompt = self.__format_prompt(instruction)

        if self.tl_support:
            toks = self.model.to_tokens(prompt).to(self.device)
            return self.model(toks, return_type="logits")
        else:
            enc = self._encode(prompt)
            return self.model(**enc).logits


    def generate(self, instruction: str, n_toks: int = 10):
        prompt = self.__format_prompt(instruction)

        if self.tl_support:
            toks = self.model.to_tokens(prompt).to(self.device)
            out_toks = self.model.generate(
                toks,
                max_new_tokens=n_toks,
                do_sample=False,
                prepend_bos=False,
            )
            gen_only = out_toks[0, toks.shape[1]:]
            return self.model.to_string(gen_only)

        else:
            enc = self._encode(prompt)
            input_len = enc["input_ids"].shape[1]

            out_ids = self.model.generate(
                **enc,
                max_new_tokens=n_toks,
                do_sample=False,
                pad_token_id=self.model.config.pad_token_id,
            )

            gen_only = out_ids[0, input_len:]
            return self.tokenizer.decode(gen_only, skip_special_tokens=True)



    def get_logit_lens(self, instruction: str, tok_pos, tok_neg):
        prompt = self.__format_prompt(instruction)

        if self.tl_support:
            if isinstance(tok_pos, str):
                tok_pos = self.model.to_single_token(tok_pos)
            if isinstance(tok_neg, str):
                tok_neg = self.model.to_single_token(tok_neg)

            toks = self.model.to_tokens(prompt).to(self.device)
            _, cache = self.model.run_with_cache(toks)
            resid = cache.stack_activation("resid_post")[:, 0]

            direction = self.model.W_U[:, tok_pos] - self.model.W_U[:, tok_neg]
            proj = resid @ direction
            return proj[:, -1].detach().cpu().numpy()

        else:
            enc = self._encode(prompt)
            out = self.model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
            )
            h = out.hidden_states[1:]

            W_U = self.model.get_output_embeddings().weight
            direction = (W_U[tok_pos] - W_U[tok_neg]).to(h[0].device)

            return np.array([
                (layer_h[0, -1] @ direction).item()
                for layer_h in h
            ])


    def get_logit_for_tok(self, logits, s: str, pos: int = -1):
        if self.tl_support:
            tok_ids = self.model.to_tokens(s, prepend_bos=False)[0]
            tok_id = tok_ids[0].item()
        else:
            tok_ids = self.tokenizer.encode(s, add_special_tokens=False)
            tok_id = tok_ids[0]
        return logits[0, pos, tok_id].item()

    def get_top_next_token_str(self, logits):
        tok_id = logits[0, -1].argmax().item()
        return self.tokenizer.decode([tok_id])
    
    def run_and_extract(self, prompt: str, hook: str):
        """
        Run a forward pass on `prompt` and extract the hidden state
        at `hook` for the last token. Returns shape [d_model].
        """
        if not self.tl_support:
            raise RuntimeError("run_and_extract only supported for TransformerLens models")

        toks = self.model.to_tokens(prompt).to(self.device)

        _, cache = self.model.run_with_cache(
            toks,
            names_filter=[hook],
        )

        act = cache[hook]          # [batch, seq, d_model]
        return act[0, -1].detach().clone()


    def run_and_insert(self, prompt: str, hook: str, vector: torch.Tensor):
        """
        Run a forward pass on `prompt` while overwriting the hidden state
        at `hook` for the last token with `vector` (shape [d_model]).
        Returns logits.
        """
        if not self.tl_support:
            raise RuntimeError("run_and_insert only supported for TransformerLens models")

        if vector.ndim != 1:
            raise ValueError("vector must be shape [d_model]")

        vector = vector.to(self.device)

        def _patch(act, hook):
            # act: [batch, seq, d_model]
            act = act.clone()
            act[:, -1, :] = vector
            return act

        toks = self.model.to_tokens(prompt).to(self.device)

        with self.model.hooks(fwd_hooks=[(hook, _patch)]):
            logits = self.model(toks, return_type="logits")

        return logits

