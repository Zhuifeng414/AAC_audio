import base64
import re

from openai import OpenAI
from pydantic import BaseModel, Field

from src.phonemes import (
    PhonemeBreakdown,
    PhonemeLookupError,
    build_phoneme_breakdown,
    build_phoneme_unit,
)


class OpenAIWorkflowError(RuntimeError):
    pass


class _SoundUnitResult(BaseModel):
    text: str = Field(description="Lowercase letters from the original word for this unit.")
    spoken_text: str = Field(description="How to pronounce this unit as a sound chunk, not letter names.")
    vowel_focus: str = Field(
        default="",
        description="The single vowel group in this unit, or empty when the unit is consonant-only.",
    )


class _PhonemeSegmentationResult(BaseModel):
    normalized_word: str = Field(description="Lowercase English word that was segmented.")
    units: list[_SoundUnitResult] = Field(description="Ordered sound units that reconstruct the word.")


def _client(api_key: str) -> OpenAI:
    if not api_key:
        raise OpenAIWorkflowError("An OpenAI API key is required for this action.")
    return OpenAI(api_key=api_key)


def _normalize_single_word(text: str) -> str:
    if not text:
        return ""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    if not tokens:
        return ""
    if tokens[0] in {"a", "an", "the"} and len(tokens) > 1:
        return tokens[1]
    return tokens[0]


def detect_main_concept(image_bytes: bytes, mime_type: str, api_key: str, model: str) -> str:
    client = _client(api_key)
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.responses.create(
            model=model,
            max_output_tokens=20,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Identify the single main concrete concept in this image. "
                                "Reply with exactly one lowercase English word and nothing else."
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
        raise OpenAIWorkflowError(f"OpenAI image recognition failed: {exc}") from exc

    concept = _normalize_single_word(response.output_text)
    if not concept:
        raise OpenAIWorkflowError(
            "OpenAI returned a response, but it could not be normalized into a single word."
        )
    return concept


def segment_word_into_phonemes(word: str, api_key: str, model: str) -> PhonemeBreakdown:
    normalized = _normalize_single_word(word)
    if not normalized:
        raise PhonemeLookupError("Enter a valid English word before requesting sound-unit segmentation.")

    client = _client(api_key)

    try:
        response = client.responses.parse(
            model=model,
            temperature=0,
            max_output_tokens=200,
            text_format=_PhonemeSegmentationResult,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You segment one English word into small readable sound units for an AAC app. "
                                "Keep the original letters in order, and the concatenation of all unit texts "
                                "must exactly reconstruct the word. Each unit must contain at most one vowel "
                                "group. Consonant-only edge units are allowed when needed. "
                                "Prefer natural speakable chunks for longer words. "
                                "Examples: chat -> ch | a | t, basketball -> bas | ket | ball. "
                                "For each unit return the unit text, a spoken_text cue that says the chunk "
                                "sound rather than spelling letter names, and the unit's vowel_focus."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Segment this word into sound units: {normalized}\n"
                                "Return JSON only."
                            ),
                        }
                    ],
                },
            ],
        )
    except Exception as exc:
        raise OpenAIWorkflowError(f"OpenAI sound-unit segmentation failed: {exc}") from exc

    parsed = response.output_parsed
    if not parsed or not parsed.units:
        raise OpenAIWorkflowError(
            "OpenAI returned an empty sound-unit segmentation response for the requested word."
        )

    segmented_word = _normalize_single_word(parsed.normalized_word) or normalized
    try:
        units = [
            build_phoneme_unit(
                text=unit.text,
                spoken_text=unit.spoken_text,
                vowel_focus=unit.vowel_focus,
            )
            for unit in parsed.units
        ]
        return build_phoneme_breakdown(
            word=segmented_word,
            phoneme_units=units,
            source=f"OpenAI {model}",
        )
    except PhonemeLookupError as exc:
        raise OpenAIWorkflowError(f"OpenAI returned an invalid sound-unit segmentation: {exc}") from exc


def synthesize_speech(
    text: str,
    api_key: str,
    model: str,
    voice: str,
    speed: float = 1.0,
    instructions: str | None = None,
    response_format: str = "wav",
) -> bytes:
    client = _client(api_key)
    kwargs = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
        "speed": speed,
    }
    if instructions:
        kwargs["instructions"] = instructions

    try:
        response = client.audio.speech.create(**kwargs)
        return response.read()
    except Exception as exc:
        if instructions:
            try:
                retry_response = client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format=response_format,
                    speed=speed,
                )
                return retry_response.read()
            except Exception:
                pass
        raise OpenAIWorkflowError(f"OpenAI speech generation failed: {exc}") from exc
