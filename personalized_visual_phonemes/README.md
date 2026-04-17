# AAC Sound Studio

Streamlit interface for:

- uploading an image and extracting the main concept as one word with OpenAI vision
- segmenting the word into readable sound units with OpenAI
- generating audio for the word and the individual sound units with OpenAI TTS
- cloning a selected or uploaded voice reference with the bundled CorentinJ SV2TTS project

## Run the app

1. Create and activate a Python environment for the Streamlit app.
2. Install the app dependencies:

```bash
pip install -r requirements.txt
```

3. Provide an API key at runtime:

```bash
export OPENAI_API_KEY="your-key-here"
```

4. Start Streamlit:

```bash
streamlit run app.py
```

## Voice cloning runtime

The bundled `Real-Time-Voice-Cloning` project targets Python `>=3.9,<3.10`, while the current
workspace uses Python `3.11`. To avoid breaking the app runtime, the voice clone action is executed
through `uv` in a separate Python 3.9 inference environment.

The clone launcher also strips Conda and `LD_LIBRARY_PATH` variables before starting the isolated
runtime, because inherited Torch/CUDA library paths from the parent shell can break the voice-clone
environment.

The first voice-clone run can take time because `uv` may need to:

- download Python 3.9
- resolve and install the minimal voice-cloning inference dependencies
- download the pretrained encoder, synthesizer, and vocoder weights

You can pre-warm that environment with:

```bash
./scripts/bootstrap_voice_clone.sh
```

## Notes

- The OpenAI key from the task file was intentionally not written into the repo.
- The segmenter now uses OpenAI to return readable sound units that preserve the letters of the
  word in order. The app then generates audio for the full word and each unit separately.
- The voice clone tab includes bundled sample voices from the upstream repository and also accepts
  uploaded audio files.
