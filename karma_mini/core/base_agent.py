"""
Base agent class for the KARMA mini framework.
"""

from typing import Dict, List, Optional
import time
import logging
from openai import OpenAI
import json

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, client: OpenAI, model_name: str, system_prompt: str = ""):
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _make_llm_call(self, prompt: str, temperature: float = 0.1, system_prompt: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Use KIT toolbox parameter logic or general OpenAI logic
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"{self.__class__.__name__} LLM call failed: {str(e)}")
            return "[]" # return empty JSON array on error

    def _parse_json_response(self, response: str) -> List[Dict]:
        try:
            # Try to find JSON array in response
            if "[" in response and "]" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                return json.loads(json_str)

            # Try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return []
