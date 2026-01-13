import json
import logging
from typing import Any, List

import requests
from decouple import config

logger = logging.getLogger(__name__)


def _get_vlm_config():
    """Get VLM configuration, supporting both Azure and local (Ollama/OpenAI) endpoints.

    Configuration is determined by environment variables:
    - Azure mode: When AZURE_OPENAI_API_KEY is set and KH_VLM_MODEL is not set
    - Local mode: When KH_VLM_MODEL is set (for Ollama, vLLM, LM Studio, etc.)

    Returns:
        dict with 'headers' and 'model' keys
    """
    azure_api_key = config("AZURE_OPENAI_API_KEY", default="")
    local_vlm_model = config("KH_VLM_MODEL", default="")

    if azure_api_key and not local_vlm_model:
        # Azure mode - model is embedded in endpoint URL
        return {
            "headers": {"Content-Type": "application/json", "api-key": azure_api_key},
            "model": None,
        }
    else:
        # Local/OpenAI-compatible mode (Ollama, vLLM, LM Studio, etc.)
        api_key = config("OPENAI_API_KEY", default="ollama")
        return {
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            "model": local_vlm_model or "gpt-4o",
        }


def generate_gpt4v(
    endpoint: str,
    images: str | List[str],
    prompt: str,
    max_tokens: int = 512,
    max_images: int = 10,
) -> str:
    vlm_config = _get_vlm_config()
    headers = vlm_config["headers"]

    if isinstance(images, str):
        images = [images]

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                    for image in images[:max_images]
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    # Add model for non-Azure endpoints (OpenAI-compatible APIs require it)
    if vlm_config["model"]:
        payload["model"] = vlm_config["model"]

    if len(images) > max_images:
        print(f"Truncated to {max_images} images (original {len(images)} images")

    response = requests.post(endpoint, headers=headers, json=payload)

    try:
        response.raise_for_status()
    except Exception as e:
        logger.exception(f"Error generating gpt4v: {response.text}; error {e}")
        return ""

    output = response.json()
    output = output["choices"][0]["message"]["content"]
    return output


def stream_gpt4v(
    endpoint: str,
    images: str | List[str],
    prompt: str,
    max_tokens: int = 512,
    max_images: int = 10,
) -> Any:
    vlm_config = _get_vlm_config()
    headers = vlm_config["headers"]

    if isinstance(images, str):
        images = [images]

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                    for image in images[:max_images]
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "logprobs": True,
        "temperature": 0,
    }

    # Add model for non-Azure endpoints (OpenAI-compatible APIs require it)
    if vlm_config["model"]:
        payload["model"] = vlm_config["model"]

    if len(images) > max_images:
        print(f"Truncated to {max_images} images (original {len(images)} images")
    try:
        response = requests.post(endpoint, headers=headers, json=payload, stream=True)
        assert response.status_code == 200, str(response.content)
        output = ""
        logprobs = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"\xef\xbb\xbf"):
                    line = line[9:]
                else:
                    line = line[6:]
                try:
                    if line == "[DONE]":
                        break
                    line = json.loads(line.decode("utf-8"))
                except Exception:
                    break
                if len(line["choices"]):
                    if line["choices"][0].get("logprobs") is None:
                        _logprobs = []
                    else:
                        _logprobs = [
                            logprob["logprob"]
                            for logprob in line["choices"][0]["logprobs"].get(
                                "content", []
                            )
                        ]

                    output += line["choices"][0]["delta"].get("content", "")
                    logprobs += _logprobs
                    yield line["choices"][0]["delta"].get("content", ""), _logprobs

    except Exception as e:
        logger.error(f"Error streaming gpt4v {e}")
        logprobs = []
        output = ""

    return output, logprobs
