import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.openai_utils import (
    OpenAIWorkflowError,
    detect_main_concept,
    segment_word_into_phonemes,
    synthesize_speech,
)
from src.phonemes import PhonemeBreakdown, PhonemeLookupError
from src.voice_clone import (
    VoiceCloneError,
    VoiceStyle,
    bundled_voice_styles,
    clone_voice_with_uv,
    describe_voice_clone_runtime,
)


APP_ROOT = Path(__file__).resolve().parent


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f4efe7;
            --ink: #1d1d1d;
            --muted: #5c564e;
            --card: rgba(255, 252, 246, 0.88);
            --line: rgba(29, 29, 29, 0.12);
            --accent: #0c8c6b;
            --accent-soft: rgba(12, 140, 107, 0.14);
            --signal: #ff6f4d;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 111, 77, 0.12), transparent 32%),
                radial-gradient(circle at top right, rgba(12, 140, 107, 0.16), transparent 26%),
                linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
            color: var(--ink);
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        code {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .hero-shell {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.6rem 1.8rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.84), rgba(255, 248, 240, 0.9)),
                linear-gradient(90deg, rgba(255, 111, 77, 0.08), rgba(12, 140, 107, 0.08));
            box-shadow: 0 18px 50px rgba(29, 29, 29, 0.08);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 3.6rem);
            line-height: 1.02;
            font-weight: 700;
            margin: 0.8rem 0 0.6rem 0;
        }

        .hero-copy {
            max-width: 54rem;
            font-size: 1rem;
            color: var(--muted);
            margin: 0;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .mini-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }

        .mini-card strong {
            display: block;
            margin-bottom: 0.25rem;
        }

        .mini-card span {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .section-card {
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--card);
            padding: 1.1rem 1.2rem;
            box-shadow: 0 12px 34px rgba(29, 29, 29, 0.05);
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin: 0.9rem 0 0.4rem 0;
        }

        .phone-chip {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            padding: 0.75rem 0.85rem;
            border-radius: 18px;
            min-width: 92px;
            background: rgba(12, 140, 107, 0.08);
            border: 1px solid rgba(12, 140, 107, 0.18);
        }

        .phone-chip strong {
            font-size: 1.15rem;
        }

        .phone-chip small {
            color: var(--muted);
            font-size: 0.76rem;
        }

        .word-banner {
            border-left: 4px solid var(--signal);
            background: rgba(255, 111, 77, 0.08);
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-top: 0.8rem;
        }

        .word-banner strong {
            font-size: 1.4rem;
        }

        .viz-shell {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .viz-card {
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.76);
            padding: 1rem;
        }

        .viz-label {
            font-size: 0.78rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .letter-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .letter-chip {
            width: 52px;
            height: 52px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 111, 77, 0.1);
            border: 1px solid rgba(255, 111, 77, 0.2);
            font-size: 1.35rem;
            font-weight: 700;
            text-transform: lowercase;
        }

        .segmented-line {
            font-size: clamp(1.4rem, 2.6vw, 2.2rem);
            font-weight: 700;
            line-height: 1.2;
        }

        .style-card {
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.78);
            padding: 0.95rem;
            min-height: 180px;
        }

        .style-family {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .style-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .style-copy {
            color: var(--muted);
            font-size: 0.88rem;
            min-height: 56px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <span class="hero-kicker">AAC Sound Studio</span>
            <h1 class="hero-title">From image to word, sound units, audio, and cloned voice.</h1>
            <p class="hero-copy">
                Upload an image, extract the main concept with OpenAI vision, break the word into
                sound units, hear each chunk, then synthesize a new sentence in a selected sample
                voice or an uploaded reference voice.
            </p>
            <div class="mini-grid">
                <div class="mini-card">
                    <strong>Vision</strong>
                    <span>One-word concept detection from an uploaded image</span>
                </div>
                <div class="mini-card">
                    <strong>Sound Units</strong>
                    <span>Readable word chunks with at most one vowel group each</span>
                </div>
                <div class="mini-card">
                    <strong>Audio</strong>
                    <span>OpenAI speech for the full word and each sound unit</span>
                </div>
                <div class="mini-card">
                    <strong>Voice Clone</strong>
                    <span>SV2TTS-based reference voice synthesis via isolated runtime</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_phoneme_chips(phoneme_breakdown) -> None:
    chips = "".join(
        (
            "<div class='phone-chip'>"
            f"<strong>{phone.text}</strong>"
            f"<small>{phone.hint}</small>"
            f"<small>say: {phone.spoken_text}</small>"
            "</div>"
        )
        for phone in phoneme_breakdown.phonemes
    )
    st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)


def render_phoneme_visualization(phoneme_breakdown) -> None:
    letters = "".join(
        f"<div class='letter-chip'>{character}</div>" for character in phoneme_breakdown.word
    )
    st.markdown(
        f"""
        <div class="viz-shell">
            <div class="viz-card">
                <div class="viz-label">Word</div>
                <div class="letter-row">{letters}</div>
            </div>
            <div class="viz-card">
                <div class="viz-label">Segmented Sound Units</div>
                <div class="segmented-line">{phoneme_breakdown.display}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_style_library(styles: list[VoiceStyle]) -> None:
    if not styles:
        st.warning("No bundled voice styles were found in the cloned repository.")
        return

    st.markdown("**Built-in voice styles**")
    columns = st.columns(2)
    for idx, style in enumerate(styles):
        with columns[idx % 2]:
            st.markdown(
                f"""
                <div class="style-card">
                    <div class="style-family">{style.family}</div>
                    <div class="style-title">{style.label}</div>
                    <div class="style-copy">{style.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.audio(style.path.read_bytes())


def require_api_key(api_key: str) -> bool:
    if api_key:
        return True
    st.error("Add an OpenAI API key in the sidebar before using image recognition, sound-unit analysis, or speech.")
    return False


def generate_audio_assets(
    phoneme_breakdown: PhonemeBreakdown,
    api_key: str,
    speech_model: str,
    tts_voice: str,
    tts_speed: float,
) -> tuple[bytes, list[tuple[str, bytes]]]:
    word_audio = synthesize_speech(
        text=phoneme_breakdown.word,
        api_key=api_key,
        model=speech_model,
        voice=tts_voice,
        speed=tts_speed,
        instructions="Pronounce the word naturally, clearly, and with no extra commentary.",
    )
    phoneme_audio = [
        (
            phone.text,
            synthesize_speech(
                text=phone.spoken_text,
                api_key=api_key,
                model=speech_model,
                voice=tts_voice,
                speed=tts_speed,
                instructions=(
                    f"Pronounce only the sound unit '{phone.text}' from the word '{phoneme_breakdown.word}'. "
                    "Do not spell letter names and do not say the full word."
                ),
            ),
        )
        for phone in phoneme_breakdown.phonemes
    ]
    return word_audio, phoneme_audio


def reset_word_outputs() -> None:
    st.session_state.word_audio = None
    st.session_state.phoneme_audio = None
    st.session_state.phoneme_breakdown = None


def sync_analysis_word_from_working_word() -> None:
    st.session_state.analysis_word = st.session_state.recognized_word
    reset_word_outputs()


def main() -> None:
    st.set_page_config(
        page_title="AAC Sound Studio",
        page_icon="🗣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    render_hero()

    if "recognized_word" not in st.session_state:
        st.session_state.recognized_word = "dog"
    if "word_audio" not in st.session_state:
        st.session_state.word_audio = None
    if "phoneme_audio" not in st.session_state:
        st.session_state.phoneme_audio = None
    if "phoneme_breakdown" not in st.session_state:
        st.session_state.phoneme_breakdown = None
    if "analysis_word" not in st.session_state:
        st.session_state.analysis_word = st.session_state.recognized_word
    if "clone_audio" not in st.session_state:
        st.session_state.clone_audio = None
    if "selected_voice_style" not in st.session_state:
        st.session_state.selected_voice_style = None

    with st.sidebar:
        st.header("Runtime")
        default_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API key", value=default_key, type="password")
        vision_model = st.text_input("Vision model", value="gpt-4.1-mini")
        phoneme_model = st.text_input("Segmentation model", value="gpt-4.1-mini")
        speech_model = st.text_input("Speech model", value="gpt-4o-mini-tts")
        tts_voice = st.text_input("OpenAI TTS voice", value="alloy")
        tts_speed = st.slider("Speech speed", min_value=0.75, max_value=1.25, value=1.0, step=0.05)
        st.caption(describe_voice_clone_runtime())

    tab_vision, tab_phonemes, tab_clone = st.tabs(
        ["1. Concept Capture", "2. Sound Unit Lab", "3. Voice Studio"]
    )

    with tab_vision:
        with st.container(border=True):
            st.subheader("Upload an image and extract the main concept")
            image_file = st.file_uploader(
                "Image input",
                type=["png", "jpg", "jpeg", "webp"],
                help="OpenAI vision returns one lowercase concept word.",
            )
            left, right = st.columns([1.1, 0.9], gap="large")
            with left:
                if image_file:
                    preview = Image.open(image_file)
                    st.image(preview, use_container_width=True)
            with right:
                if st.button("Recognize Main Concept", type="primary", use_container_width=True):
                    if image_file and require_api_key(api_key):
                        try:
                            with st.spinner("Reading image with OpenAI vision..."):
                                concept = detect_main_concept(
                                    image_bytes=image_file.getvalue(),
                                    mime_type=image_file.type or "image/jpeg",
                                    api_key=api_key,
                                    model=vision_model,
                                )
                            st.session_state.recognized_word = concept
                            st.session_state.analysis_word = concept
                            reset_word_outputs()
                            st.success(f"Main concept detected: {concept}")
                        except OpenAIWorkflowError as exc:
                            st.error(str(exc))
                    elif not image_file:
                        st.warning("Upload an image first.")

                st.text_input(
                    "Working word",
                    key="recognized_word",
                    help="You can manually override the detected word before moving to sound-unit analysis.",
                    on_change=sync_analysis_word_from_working_word,
                )

                st.markdown(
                    f"""
                    <div class="word-banner">
                        <div>Current target word</div>
                        <strong>{st.session_state.recognized_word}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Generate Word Audio", use_container_width=True):
                    if require_api_key(api_key):
                        try:
                            with st.spinner("Generating word audio..."):
                                st.session_state.word_audio = synthesize_speech(
                                    text=st.session_state.recognized_word,
                                    api_key=api_key,
                                    model=speech_model,
                                    voice=tts_voice,
                                    speed=tts_speed,
                                    instructions=(
                                        "Pronounce the word naturally, clearly, and with no extra commentary."
                                    ),
                                )
                            st.success("Word audio generated.")
                        except OpenAIWorkflowError as exc:
                            st.error(str(exc))

                if st.session_state.word_audio:
                    st.audio(st.session_state.word_audio, format="audio/wav")

            st.markdown("### Word And Sound-Unit Segmentation")
            st.caption("The whole word and its segmented sound units appear here after OpenAI analysis.")
            quick_breakdown = st.session_state.phoneme_breakdown
            if quick_breakdown and quick_breakdown.word == st.session_state.recognized_word:
                render_phoneme_visualization(quick_breakdown)
                render_phoneme_chips(quick_breakdown)
                st.code(
                    f"{quick_breakdown.word} -> {quick_breakdown.display}",
                    language=None,
                )
                st.caption(f"Segmentation source: {quick_breakdown.source}")
            else:
                st.info(
                    "Analyze the current word in the Sound Unit Lab to fetch its OpenAI segmentation."
                )

    with tab_phonemes:
        with st.container(border=True):
            st.subheader("Segment the word into sound units and hear every chunk")
            current_word = st.text_input("Word to analyze", key="analysis_word")
            phoneme_breakdown = (
                st.session_state.phoneme_breakdown
                if st.session_state.phoneme_breakdown
                and st.session_state.phoneme_breakdown.word == current_word.strip().lower()
                else None
            )

            if st.button("Analyze With OpenAI", use_container_width=True):
                if require_api_key(api_key):
                    try:
                        with st.spinner("Segmenting word into sound units with OpenAI..."):
                            phoneme_breakdown = segment_word_into_phonemes(
                                word=current_word,
                                api_key=api_key,
                                model=phoneme_model,
                            )
                        st.session_state.phoneme_breakdown = phoneme_breakdown
                        st.session_state.word_audio = None
                        st.session_state.phoneme_audio = None
                    except (OpenAIWorkflowError, PhonemeLookupError) as exc:
                        st.session_state.phoneme_breakdown = None
                        phoneme_breakdown = None
                        st.error(str(exc))

            if phoneme_breakdown:
                render_phoneme_visualization(phoneme_breakdown)
                render_phoneme_chips(phoneme_breakdown)
                st.code(phoneme_breakdown.display, language=None)
                st.caption(f"Segmentation source: {phoneme_breakdown.source}")
            else:
                st.info("Run OpenAI sound-unit analysis for the current word to preview its segmentation.")

            if phoneme_breakdown and st.button(
                "Generate Word + Unit Audio",
                type="primary",
                use_container_width=True,
            ):
                if require_api_key(api_key):
                    try:
                        with st.spinner("Generating OpenAI speech..."):
                            st.session_state.word_audio, st.session_state.phoneme_audio = generate_audio_assets(
                                phoneme_breakdown=phoneme_breakdown,
                                api_key=api_key,
                                speech_model=speech_model,
                                tts_voice=tts_voice,
                                tts_speed=tts_speed,
                            )
                    except (OpenAIWorkflowError, PhonemeLookupError) as exc:
                        st.error(str(exc))

            if st.session_state.word_audio:
                st.markdown("**Word audio**")
                st.audio(st.session_state.word_audio, format="audio/wav")

            if st.session_state.phoneme_audio:
                st.markdown("**Sound-unit audio**")
                columns = st.columns(min(4, len(st.session_state.phoneme_audio)))
                for idx, (unit_text, audio_bytes) in enumerate(st.session_state.phoneme_audio):
                    with columns[idx % len(columns)]:
                        st.caption(unit_text)
                        st.audio(audio_bytes, format="audio/wav")

    with tab_clone:
        with st.container(border=True):
            st.subheader("Clone into a selected voice style")
            st.caption(
                "This uses the bundled CorentinJ SV2TTS project through an isolated `uv` runtime. "
                "The first run can take a while because dependencies and pretrained weights may download."
            )

            clone_text = st.text_area(
                "Sentence to synthesize",
                value=f"The word is {st.session_state.recognized_word}.",
                height=100,
            )
            reference_mode = st.radio(
                "Reference voice source",
                options=["Voice style library", "Upload custom voice"],
                horizontal=True,
            )

            reference_bytes = None
            reference_name = None

            if reference_mode == "Voice style library":
                styles = bundled_voice_styles()
                style_lookup = {style.label: style for style in styles}
                if style_lookup:
                    if not st.session_state.selected_voice_style:
                        st.session_state.selected_voice_style = styles[0].label
                    style_names = list(style_lookup)
                    st.markdown("### Choose A Voice Style")
                    selected_style_label = st.selectbox(
                        "Voice style",
                        options=style_names,
                        index=(
                            style_names.index(st.session_state.selected_voice_style)
                            if st.session_state.selected_voice_style in style_lookup
                            else 0
                        ),
                    )
                    st.session_state.selected_voice_style = selected_style_label
                    selected_style = style_lookup[selected_style_label]
                    reference_name = selected_style.label
                    reference_bytes = selected_style.path.read_bytes()
                    st.markdown(
                        f"""
                        <div class="word-banner">
                            <div>Selected voice style</div>
                            <strong>{selected_style.label}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(selected_style.description)
                    st.audio(reference_bytes)
                    render_style_library(styles)
                else:
                    st.warning(
                        "The bundled voice-style library was not found. Re-clone the repo or use Upload custom voice."
                    )
            else:
                uploaded_voice = st.file_uploader(
                    "Reference audio",
                    type=["wav", "mp3", "m4a", "flac", "ogg"],
                    help="Upload a short clean sample of the voice you want to clone.",
                )
                if uploaded_voice is not None:
                    reference_name = uploaded_voice.name
                    reference_bytes = uploaded_voice.getvalue()
                    st.audio(reference_bytes)

            use_gpu = st.checkbox("Try CUDA runtime", value=False)

            if st.button("Generate Cloned Voice", type="primary", use_container_width=True):
                if not reference_bytes:
                    st.warning("Provide a reference voice sample first.")
                elif not clone_text.strip():
                    st.warning("Enter the sentence to synthesize.")
                else:
                    suffix = Path(reference_name or "reference.wav").suffix or ".wav"
                    with tempfile.TemporaryDirectory(prefix="aac_voice_clone_") as tmp_dir:
                        tmp_root = Path(tmp_dir)
                        reference_path = tmp_root / f"reference{suffix}"
                        reference_path.write_bytes(reference_bytes)
                        output_path = tmp_root / "cloned.wav"

                        try:
                            with st.spinner("Running SV2TTS voice cloning..."):
                                clone_voice_with_uv(
                                    reference_audio_path=reference_path,
                                    text=clone_text,
                                    output_path=output_path,
                                    use_gpu=use_gpu,
                                )
                            st.session_state.clone_audio = output_path.read_bytes()
                        except VoiceCloneError as exc:
                            st.error(str(exc))

            if st.session_state.clone_audio:
                st.markdown("**Cloned output**")
                st.audio(st.session_state.clone_audio, format="audio/wav")


if __name__ == "__main__":
    main()
