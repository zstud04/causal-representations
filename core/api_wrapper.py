from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
from openai import OpenAI

from google.genai import types


@dataclass
class APIResult:
    text: str
    raw: Any = None
    usage: Optional[Dict[str, Any]] = None
    reasoning: Any = None
    provider: str = ""
    model: str = ""
    response_id: Optional[str] = None


class APIWrapper:
    def __init__(self, provider: str, model: str, api_key: str, base_url: Optional[str] = None):
        self.provider = provider.lower().strip()
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            ) if self.base_url else OpenAI(api_key=self.api_key)

        elif self.provider == "deepseek":
            # DeepSeek uses OpenAI-compatible SDK
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.deepseek.com"
            )

        elif self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError("provider must be 'openai', 'deepseek', or 'gemini'")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_output_tokens: int = 256,
    ) -> APIResult:

        # -------------------------
        # OpenAI
        # -------------------------
        if self.provider == "openai":
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=int(max_output_tokens),
            )

            try:
                reasoning_text = resp.output[0].summary[0].text
            except Exception:
                reasoning_text = None

            return APIResult(
                text=resp.output_text or "",
                raw=resp,
                usage=getattr(resp, "usage", None),
                reasoning=reasoning_text,
                provider="openai",
                model=self.model_name,
                response_id=getattr(resp, "id", None),
            )

        # -------------------------
        # DeepSeek
        # -------------------------
        if self.provider == "deepseek":
            messages = (
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            )

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=int(max_output_tokens),
            )

            msg = resp.choices[0].message

            return APIResult(
                text=msg.content or "",
                raw=resp,
                usage=getattr(resp, "usage", None),
                reasoning=getattr(msg, "reasoning_content", None),
                provider="deepseek",
                model=self.model_name,
                response_id=getattr(resp, "id", None),
            )

        # -------------------------
        # Gemini (unchanged)
        # -------------------------
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=int(max_output_tokens),
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                ),
            ),
        )

        return APIResult(
            text=resp.text or "",
            raw=resp,
            usage=getattr(resp, "usage_metadata", None),
            reasoning=resp.candidates[0].content.parts[0].text,
            provider="gemini",
            model=self.model_name,
            response_id=getattr(resp, "id", None),
        )
