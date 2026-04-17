import re
from dataclasses import dataclass


_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


class PhonemeLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class PhonemeUnit:
    text: str
    vowel_focus: str
    hint: str
    spoken_text: str


@dataclass(frozen=True)
class PhonemeBreakdown:
    word: str
    phonemes: list[PhonemeUnit]
    display: str
    source: str


def _normalize_word(word: str) -> str:
    return re.sub(r"[^A-Za-z'-]", "", word or "").lower().strip()


def _vowel_groups(text: str) -> list[str]:
    return _VOWEL_GROUP_RE.findall(text)


def build_phoneme_unit(text: str, spoken_text: str, vowel_focus: str = "") -> PhonemeUnit:
    normalized_text = _normalize_word(text)
    if not normalized_text:
        raise PhonemeLookupError("OpenAI returned an empty segmentation unit.")

    groups = _vowel_groups(normalized_text)
    if len(groups) > 1:
        raise PhonemeLookupError(
            f"Invalid unit '{normalized_text}'. Each unit must contain at most one vowel group."
        )

    normalized_focus = _normalize_word(vowel_focus)
    if groups:
        focus = normalized_focus or groups[0].lower()
        hint = f"vowel focus: {focus}"
    else:
        focus = ""
        hint = "consonant-only edge unit"

    spoken = (spoken_text or normalized_text).strip()
    return PhonemeUnit(
        text=normalized_text,
        vowel_focus=focus,
        hint=hint,
        spoken_text=spoken,
    )


def build_phoneme_breakdown(
    word: str,
    phoneme_units: list[PhonemeUnit],
    source: str = "OpenAI sound-unit segmentation",
) -> PhonemeBreakdown:
    normalized = _normalize_word(word)
    if not normalized:
        raise PhonemeLookupError("Enter a valid English word before requesting segmentation.")
    if not phoneme_units:
        raise PhonemeLookupError("No segmentation units were returned for the requested word.")

    combined = "".join(unit.text for unit in phoneme_units)
    if combined != normalized:
        raise PhonemeLookupError(
            f"OpenAI returned units that do not reconstruct the word: '{combined}' != '{normalized}'."
        )

    if not any(_vowel_groups(unit.text) for unit in phoneme_units):
        raise PhonemeLookupError("The returned segmentation does not contain any vowel-bearing unit.")

    display = " ".join(unit.text for unit in phoneme_units)
    return PhonemeBreakdown(
        word=normalized,
        phonemes=phoneme_units,
        display=display,
        source=source,
    )
