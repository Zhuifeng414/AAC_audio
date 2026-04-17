# Eye Focus Tracker

This Streamlit app turns a laptop webcam into a rough eye-focus pointer for the current display. It:

- tracks iris landmarks from the webcam with MediaPipe
- estimates a gaze point on the screen
- draws a red focus box and gaze dot on a live screen preview
- crops the focused region
- sends that crop to the OpenAI API for a short concept label

## What it is good for

- fast local prototyping for gaze-assisted AAC ideas
- showing where the user is probably looking on the screen
- attaching a short concept label to that focus region

## Limits

- this is an approximate gaze mapper, not clinical eye tracking
- accuracy depends heavily on camera position, lighting, and calibration quality
- on macOS you must grant both Camera and Screen Recording permissions

## Setup

```bash
cd /home/tuq24452/code/GUIAgent/AAC_26/eye_focus_tracker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
streamlit run app.py
```

## Suggested flow

1. Start the tracker.
2. Look at each calibration target and click the matching button.
3. Watch the red box move on the screen preview.
4. Let the app refresh the crop label every few seconds.

## Notes for MacBook Pro

- If the webcam does not open, check `Camera` permission in macOS Settings.
- If the screen preview is black, check `Screen Recording` permission for the terminal or Python app you launched.
- For Apple Silicon, a Python 3.11 virtualenv is the safest default if package wheels vary across versions.
