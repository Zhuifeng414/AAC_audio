import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SV2TTS voice cloning in an isolated env.")
    parser.add_argument("--project-root", required=True, help="Main app project root.")
    parser.add_argument("--reference", required=True, help="Reference audio file path.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", required=True, help="Where to write the cloned wav file.")
    parser.add_argument("--allow-gpu", action="store_true", help="Allow the worker to use CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    voice_clone_root = project_root / "third_party" / "Real-Time-Voice-Cloning"
    models_root = voice_clone_root / "saved_models"

    if str(voice_clone_root) not in sys.path:
        sys.path.insert(0, str(voice_clone_root))

    from utils.default_models import ensure_default_models
    from encoder import inference as encoder
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder

    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.allow_gpu and os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ensure_default_models(models_root)

    encoder_path = models_root / "default" / "encoder.pt"
    synthesizer_path = models_root / "default" / "synthesizer.pt"
    vocoder_path = models_root / "default" / "vocoder.pt"

    if not encoder.is_loaded():
        encoder.load_model(encoder_path, device="cpu")

    synthesizer = Synthesizer(synthesizer_path, verbose=False)
    if not vocoder.is_loaded():
        vocoder.load_model(vocoder_path, verbose=False)

    preprocessed_wav = encoder.preprocess_wav(reference_path)
    embedding = encoder.embed_utterance(preprocessed_wav)
    spectrogram = synthesizer.synthesize_spectrograms([args.text], [embedding])[0]
    generated_wav = vocoder.infer_waveform(spectrogram)

    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)

    buffer = io.BytesIO()
    sf.write(buffer, generated_wav.astype(np.float32), synthesizer.sample_rate, format="WAV")
    output_path.write_bytes(buffer.getvalue())


if __name__ == "__main__":
    main()
