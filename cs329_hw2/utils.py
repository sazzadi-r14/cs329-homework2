import copy
import requests
import time
import os
import openai
import anthropic
from groq import Groq
import google.generativeai as google_genai
import json
import random
from litellm import completion
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def generate_together(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    output = None
    key = os.environ.get("TOGETHER_API_KEY")
    
    for attempt, sleep_time in enumerate([1, 2, 4, 8, 16, 32], 1):
        res = None
        try:
            endpoint = "https://api.together.xyz/v1/chat/completions"
            time.sleep(2)

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {key}",
                },
            )

            output = res.json()["choices"][0]["message"]["content"]
            break

        except Exception as e:
            if attempt == 6:  # Last attempt
                raise e
            time.sleep(sleep_time)

    return output.strip() if output else None


def generate_openai(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=key)

    if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        messages = [msg for msg in messages if msg["role"] != "system"]

    for attempt, sleep_time in enumerate([1, 2, 4, 8, 16, 32, 64], 1):
        try:
            if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            if attempt == 7:  # Last attempt
                raise e
            time.sleep(sleep_time)


def generate_anthropic(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Client(api_key=key)
    
    # Extract system message once outside the retry loop
    system = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
    messages_alt = [msg for msg in messages if msg["role"] != "system"]
    
    for attempt, sleep_time in enumerate([1, 2, 4, 8, 16, 32, 64], 1):
        try:
            completion = client.messages.create(
                model=model,
                system=system,
                messages=messages_alt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.content[0].text.strip()
            
        except Exception as e:
            if attempt == 7:  # Last attempt
                raise e
            time.sleep(sleep_time)