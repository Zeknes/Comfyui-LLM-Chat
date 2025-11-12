#
# Author: Zeknes
# Repo: https://github.com/Zeknes/ComfyUI-LLMChat
#

import torch
import numpy as np
from PIL import Image
import base64
import io
import requests
import json
import time

class LLMChatNode:
    """
    A unified custom node for interacting with OpenAI and Ollama LLMs.
    """

    def __init__(self):
        self.ollama_models = self._get_ollama_models("http://127.0.0.1:11434")

    def _get_ollama_models(self, base_url: str):
        try:
            url = f"{base_url.rstrip('/')}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models] if models else ["default (is Ollama running?)"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return ["default (error fetching models)"]

    @classmethod
    def INPUT_TYPES(cls):
        # We instantiate the class to get the model list
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
                # Prompts are now at the bottom as requested
                "system_prompt": ("STRING", {"default": "You are a image description generation expert. your description is in natural language, no markdown or any other format. You output the description directly, no explaination", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Describe the image.", "multiline": True}),
            },
            "optional": {"image": ("IMAGE",)}
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "LLMChat"

    def _call_openai(self, base_url, api_key, model, messages, seed, max_retries):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": messages, "seed": seed}
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=data, timeout=20)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_exception = e
                print(f"OpenAI Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return f"Error: OpenAI failed after {max_retries} retries. Last error: {last_exception}"

    def _call_ollama(self, base_url, model, messages, seed, max_retries):
        data = {"model": model, "messages": messages, "stream": False, "options": {"seed": seed}}
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=data, timeout=20)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                last_exception = e
                print(f"Ollama Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return f"Error: Ollama failed after {max_retries} retries. Last error: {last_exception}"

    def execute(self, mode, openai_base_url, api_key, openai_model, ollama_base_url, ollama_model, seed, max_retries, system_prompt, user_prompt, image=None):
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

        # Prepare messages for OpenAI format
        openai_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        if img_base64:
            openai_messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})

        # Prepare messages for Ollama format
        ollama_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt, "images": [img_base64] if img_base64 else []}]

        final_result = ""
        if mode == "OpenAI":
            final_result = self._call_openai(openai_base_url, api_key, openai_model, openai_messages, seed, max_retries)
        
        elif mode == "Ollama":
            final_result = self._call_ollama(ollama_base_url, ollama_model, ollama_messages, seed, max_retries)

        elif mode == "Smart":
            print("Smart Mode: Trying OpenAI first...")
            final_result = self._call_openai(openai_base_url, api_key, openai_model, openai_messages, seed, max_retries)
            if "error" in final_result.lower():
                print("OpenAI failed. Smart Mode: Falling back to Ollama...")
                final_result = self._call_ollama(ollama_base_url, ollama_model, ollama_messages, seed, max_retries)

        if "error" in final_result.lower():
            raise Exception(final_result)

        return (final_result,)


NODE_CLASS_MAPPINGS = {
    "LLMChat": LLMChatNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMChat": "LLM Chat"
}
