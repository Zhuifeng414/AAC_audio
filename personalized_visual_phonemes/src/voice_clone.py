import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
VOICE_CLONE_ROOT = APP_ROOT / "third_party" / "Real-Time-Voice-Cloning"
WORKER_PATH = APP_ROOT / "src" / "voice_clone_worker.py"
VOICE_CLONE_RUNTIME_ROOT = APP_ROOT / "voice_clone_runtime"

_SAMPLE_LABELS = {
    "British VCTK p240": VOICE_CLONE_ROOT / "samples" / "p240_00000.mp3",
    "British VCTK p260": VOICE_CLONE_ROOT / "samples" / "p260_00000.mp3",
    "LibriSpeech 1320": VOICE_CLONE_ROOT / "samples" / "1320_00000.mp3",
    "LibriSpeech 3575": VOICE_CLONE_ROOT / "samples" / "3575_00000.mp3",
    "LibriSpeech 6829": VOICE_CLONE_ROOT / "samples" / "6829_00000.mp3",
    "LibriSpeech 8230": VOICE_CLONE_ROOT / "samples" / "8230_00000.mp3",
}


class VoiceCloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceStyle:
    label: str
    path: Path
    family: str
    description: str


def bundled_voice_options() -> dict[str, Path]:
    return {label: path for label, path in _SAMPLE_LABELS.items() if path.exists()}


def bundled_voice_styles() -> list[VoiceStyle]:
    styles = [
        VoiceStyle(
            label="British VCTK p240",
            path=VOICE_CLONE_ROOT / "samples" / "p240_00000.mp3",
            family="VCTK",
            description="Clean British English sample from the bundled VCTK voices.",
        ),
        VoiceStyle(
            label="British VCTK p260",
            path=VOICE_CLONE_ROOT / "samples" / "p260_00000.mp3",
            family="VCTK",
            description="Alternative British English sample with a different tone.",
        ),
        VoiceStyle(
            label="LibriSpeech 1320",
            path=VOICE_CLONE_ROOT / "samples" / "1320_00000.mp3",
            family="LibriSpeech",
            description="Audiobook-style sample voice from LibriSpeech.",
        ),
        VoiceStyle(
            label="LibriSpeech 3575",
            path=VOICE_CLONE_ROOT / "samples" / "3575_00000.mp3",
            family="LibriSpeech",
            description="Another bundled LibriSpeech sample with a different cadence.",
        ),
        VoiceStyle(
            label="LibriSpeech 6829",
            path=VOICE_CLONE_ROOT / "samples" / "6829_00000.mp3",
            family="LibriSpeech",
            description="Bundled LibriSpeech reference with a distinct reading rhythm.",
        ),
        VoiceStyle(
            label="LibriSpeech 8230",
            path=VOICE_CLONE_ROOT / "samples" / "8230_00000.mp3",
            family="LibriSpeech",
            description="Bundled LibriSpeech sample with a different timbre.",
        ),
    ]
    return [style for style in styles if style.path.exists()]


def describe_voice_clone_runtime() -> str:
    python_label = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"App runtime: Python {python_label}. Voice cloning is executed in a separate "
        "`uv`-managed Python 3.9 environment because the bundled SV2TTS repo targets 3.9."
    )


def _sanitized_clone_env(use_gpu: bool) -> dict[str, str]:
    env = dict(os.environ)
    for key in [
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_SHLVL",
        "CONDA_PROMPT_MODIFIER",
        "CONDA_PYTHON_EXE",
        "_CE_CONDA",
        "_CE_M",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
    ]:
        env.pop(key, None)
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def clone_voice_with_uv(
    reference_audio_path: Path,
    text: str,
    output_path: Path,
    use_gpu: bool = False,
) -> None:
    if not VOICE_CLONE_ROOT.exists():
        raise VoiceCloneError("The voice cloning repository is missing from `third_party/`.")
    if not VOICE_CLONE_RUNTIME_ROOT.exists():
        raise VoiceCloneError("The voice cloning runtime project is missing from `voice_clone_runtime/`.")
    if not WORKER_PATH.exists():
        raise VoiceCloneError("The voice cloning worker script is missing.")
    if shutil.which("uv") is None:
        raise VoiceCloneError("`uv` is required for voice cloning, but it was not found on PATH.")

    command = [
        "uv",
        "run",
        "--project",
        str(VOICE_CLONE_RUNTIME_ROOT),
        "--python",
        "3.9",
        "--extra",
        "cuda" if use_gpu else "cpu",
        "python",
        str(WORKER_PATH),
        "--project-root",
        str(APP_ROOT),
        "--reference",
        str(reference_audio_path),
        "--text",
        text,
        "--output",
        str(output_path),
    ]
    if use_gpu:
        command.append("--allow-gpu")

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_sanitized_clone_env(use_gpu=use_gpu),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise VoiceCloneError(f"Voice cloning failed: {stderr}") from exc

    if not output_path.exists():
        stdout = completed.stdout.strip()
        raise VoiceCloneError(
            "Voice cloning finished without creating the expected output file."
            + (f" Worker output: {stdout}" if stdout else "")
        )
