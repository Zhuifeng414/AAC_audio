import base64
import re

from openai import OpenAI


class OpenAIWorkflowError(RuntimeError):
    """Raised when the OpenAI-assisted concept recognition flow fails."""


def _client(api_key: str) -> OpenAI:
    if not api_key:
        raise OpenAIWorkflowError("An OpenAI API key is required for concept recognition.")
    return OpenAI(api_key=api_key)


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 -]+", " ", text).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return " ".join(cleaned.split()[:3])


def detect_focus_concept(image_bytes: bytes, mime_type: str, api_key: str, model: str) -> str:
    client = _client(api_key)
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.responses.create(
            model=model,
            max_output_tokens=24,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are labeling a tiny screen crop from an eye-tracking accessibility app. "
                                "Return the main concept as a very short noun phrase of at most three lowercase words. "
                                "No punctuation. No extra explanation."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{encoded}",
                        },
                    ],
                }
            ],
        )
    except Exception as exc:
        raise OpenAIWorkflowError(f"OpenAI concept recognition failed: {exc}") from exc

    label = _normalize_label(response.output_text or "")
    if not label:
        raise OpenAIWorkflowError("OpenAI returned an empty concept label.")
    return label
