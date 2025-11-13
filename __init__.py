#
# Author: Zeknes (Original), with async and multi-output modifications
# Repo: https://github.com/Zeknes/ComfyUI-LLMChat
# Version: 3.0 (Async + Multi-output)
#
# This version uses aiohttp for asynchronous network requests and
# splits the AI's output into three distinct parts:
# 1. thinking: Content within <think>...</think> tags.
# 2. result: Content outside the tags.
# 3. response: The full, raw output from the AI.
#

import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import aiohttp
import asyncio
import re

class LLMChatNode:
    """
    An asynchronous, multi-output custom node for interacting with OpenAI and Ollama LLMs.
    """

    def __init__(self):
        self.ollama_models = self._get_ollama_models_sync("http://127.0.0.1:11434")

    def _get_ollama_models_sync(self, base_url: str):
        try_url = f"{base_url.rstrip('/')}/api/tags"
        try:
            response = requests.get(try_url, timeout=3)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models] if models else ["default (is Ollama running?)"]
        except requests.exceptions.RequestException:
            return ["default (error fetching models)"]

    @classmethod
    def INPUT_TYPES(cls):
        instance = cls()
        return {
            "required": {
                "mode": (["OpenAI", "Ollama", "Smart"],),
                "openai_base_url": ("STRING", {"default": "https://openrouter.ai/api/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "openai_model": ("STRING", {"default": "google/gemini-2.5-flash"}),
                "ollama_base_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (instance.ollama_models,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "system_prompt": ("STRING", {"default": "You are an image description generation expert. Your description is in natural language, no markdown or any other format. You output the description directly, no explanation. If you need to think, use <think>your thoughts here</think> tags.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe the image.", "multiline": True}),
            },
            "optional": {"image": ("IMAGE",)}
        }

    # V3 Change: Updated return types and names
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("thinking", "result", "response")
    FUNCTION = "execute"
    CATEGORY = "LLMChat"

    @staticmethod
    def _parse_response(raw_response: str):
        """
        Parses the raw AI response to extract 'thinking' and 'result' parts.
        
        Args:
            raw_response: The full string output from the language model.
            
        Returns:
            A tuple containing (thinking_text, result_text).
        """
        thinking_text = ""
        # Use regex to find all occurrences of the think block
        # re.DOTALL allows '.' to match newlines
        matches = list(re.finditer(r"<think>(.*?)</think>", raw_response, re.DOTALL))
        
        if matches:
            # Join the content of all think blocks
            thinking_parts = [match.group(1).strip() for match in matches]
            thinking_text = "\n\n".join(thinking_parts)
            
            # The result is the raw response with all think blocks removed
            result_text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
        else:
            # No think tags found, so the entire response is the result
            thinking_text = ""
            result_text = raw_response.strip()
            
        return thinking_text, result_text

    async def _call_api_async(self, session, url, headers, data, max_retries, service_name):
        last_exception = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=data, timeout=60) as response: # Increased timeout
                    response.raise_for_status()
                    response_json = await response.json()
                    
                    if service_name == "OpenAI":
                        if "choices" not in response_json or not response_json["choices"]:
                            raise KeyError("API response is missing 'choices' field or it's empty.")
                        return response_json["choices"][0]["message"]["content"]
                    elif service_name == "Ollama":
                        return response_json["message"]["content"]
                    
            except Exception as e:
                last_exception = e
                print(f"{service_name} Attempt {attempt + 1}/{max_retries} failed: {e}")
                if isinstance(e, aiohttp.ClientResponseError):
                     try:
                        error_text = await response.text()
                        print(f"Full API Error Response: {error_text}")
                     except:
                        pass
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        return f"Error: {service_name} failed after {max_retries} retries. Last error: {last_exception}"

    async def execute(self, mode, openai_base_url, api_key, openai_model, ollama_base_url, ollama_model, seed, max_retries, system_prompt, user_prompt, image=None):
        img_base64 = None
        if image is not None:
            try:
                i = 255. * image.cpu().numpy().squeeze()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e:
                raise Exception(f"Error processing image: {e}")

        openai_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        if img_base64:
            openai_messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

        ollama_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt, "images": [img_base64] if img_base64 else []}]

        raw_response = ""
        
        async with aiohttp.ClientSession() as session:
            if mode == "OpenAI":
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {"model": openai_model, "messages": openai_messages, "seed": seed}
                url = f"{openai_base_url.rstrip('/')}/chat/completions"
                raw_response = await self._call_api_async(session, url, headers, data, max_retries, "OpenAI")
            
            elif mode == "Ollama":
                data = {"model": ollama_model, "messages": ollama_messages, "stream": False, "options": {"seed": seed}}
                url = f"{ollama_base_url.rstrip('/')}/api/chat"
                raw_response = await self._call_api_async(session, url, None, data, max_retries, "Ollama")

            elif mode == "Smart":
                print("Smart Mode: Trying OpenAI first...")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {"model": openai_model, "messages": openai_messages, "seed": seed}
                url = f"{openai_base_url.rstrip('/')}/chat/completions"
                raw_response = await self._call_api_async(session, url, headers, data, max_retries, "OpenAI")
                
                if "error" in raw_response.lower():
                    print("OpenAI failed. Smart Mode: Falling back to Ollama...")
                    data = {"model": ollama_model, "messages": ollama_messages, "stream": False, "options": {"seed": seed}}
                    url = f"{ollama_base_url.rstrip('/')}/api/chat"
                    raw_response = await self._call_api_async(session, url, None, data, max_retries, "Ollama")

        if "error" in raw_response.lower():
            raise Exception(raw_response)

        # V3 Change: Parse the response and return three distinct outputs
        thinking, result = self._parse_response(raw_response)

        return (thinking, result, raw_response)


# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "LLMChat": LLMChatNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMChat": "LLM Chat"
}
