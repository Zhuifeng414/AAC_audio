from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.runtime import EyeFocusRuntime


APP_ROOT = Path(__file__).resolve().parent


CALIBRATION_POINTS = [
    ("Top Left", (0.08, 0.12)),
    ("Top Center", (0.50, 0.12)),
    ("Top Right", (0.92, 0.12)),
    ("Mid Left", (0.08, 0.50)),
    ("Center", (0.50, 0.50)),
    ("Mid Right", (0.92, 0.50)),
    ("Bottom Left", (0.08, 0.88)),
    ("Bottom Center", (0.50, 0.88)),
    ("Bottom Right", (0.92, 0.88)),
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f2efe8;
            --ink: #151825;
            --muted: #5d6473;
            --line: rgba(21, 24, 37, 0.12);
            --card: rgba(255, 255, 255, 0.82);
            --signal: #ff4b4b;
            --accent: #007f73;
            --accent-soft: rgba(0, 127, 115, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(255, 75, 75, 0.12), transparent 24%),
                radial-gradient(circle at 92% 12%, rgba(0, 127, 115, 0.14), transparent 24%),
                linear-gradient(180deg, #fffdf8 0%, var(--bg) 100%);
            color: var(--ink);
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        code {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .shell {
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.4rem 1.5rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(255, 248, 241, 0.84));
            box-shadow: 0 16px 40px rgba(21, 24, 37, 0.08);
            margin-bottom: 1rem;
        }

        .kicker {
            display: inline-block;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            background: var(--accent-soft);
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .title {
            margin: 0.8rem 0 0.5rem 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 0.98;
            font-weight: 700;
        }

        .copy {
            color: var(--muted);
            max-width: 60rem;
            margin: 0;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .metric {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.75);
            padding: 0.9rem 1rem;
        }

        .metric strong {
            display: block;
            font-size: 1.1rem;
        }

        .metric span {
            color: var(--muted);
            font-size: 0.86rem;
        }

        .card {
            border: 1px solid var(--line);
            border-radius: 22px;
            background: var(--card);
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 34px rgba(21, 24, 37, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_runtime() -> EyeFocusRuntime:
    if "eye_focus_runtime" not in st.session_state:
        st.session_state.eye_focus_runtime = EyeFocusRuntime()
    return st.session_state.eye_focus_runtime


def render_hero(state) -> None:
    size = state.screen_size if state.screen_size != (0, 0) else ("?", "?")
    st.markdown(
        f"""
        <section class="shell">
            <span class="kicker">Desktop Eye Focus</span>
            <h1 class="title">Track gaze, crop the screen, label the concept.</h1>
            <p class="copy">
                The app reads your webcam, estimates a gaze point, draws a live red focus box on a screen preview,
                and sends the cropped region to OpenAI for a short concept label.
            </p>
            <div class="metrics">
                <div class="metric"><strong>{'Live' if state.running else 'Idle'}</strong><span>Capture loop</span></div>
                <div class="metric"><strong>{state.calibration_samples}</strong><span>Calibration samples</span></div>
                <div class="metric"><strong>{'Ready' if state.calibrated else 'Approximate'}</strong><span>Gaze mapping</span></div>
                <div class="metric"><strong>{size[0]} x {size[1]}</strong><span>Screen pixels</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Eye Focus Tracker", page_icon=":red_circle:", layout="wide")
    inject_styles()

    runtime = get_runtime()
    state = runtime.get_state()

    st_autorefresh(interval=250 if state.running else 1500, key="eye-focus-refresh")

    with st.sidebar:
        st.header("Controls")
        camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
        crop_size = st.slider("Focus crop size", min_value=120, max_value=420, value=240, step=20)
        recognition_interval = st.slider(
            "Recognition cadence (seconds)",
            min_value=1.0,
            max_value=6.0,
            value=2.2,
            step=0.2,
        )
        api_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="If empty, the app still tracks gaze but skips concept labeling.",
        )
        model = st.text_input("Vision model", value="gpt-4.1-mini")

        runtime.configure(
            camera_index=int(camera_index),
            crop_size=int(crop_size),
            recognition_interval=float(recognition_interval),
            openai_api_key=api_key,
            openai_model=model,
        )

        button_left, button_right = st.columns(2)
        with button_left:
            if st.button("Start tracker", use_container_width=True):
                runtime.start()
                st.rerun()
        with button_right:
            if st.button("Stop tracker", use_container_width=True):
                runtime.stop()
                st.rerun()

        if st.button("Clear calibration", use_container_width=True):
            runtime.clear_calibration()
            st.rerun()

        st.caption(
            "On macOS, grant both Camera and Screen Recording permissions to Terminal or the Python app."
        )

    state = runtime.get_state()
    render_hero(state)

    st.markdown('<section class="card">', unsafe_allow_html=True)
    st.subheader("Calibration")
    st.write(
        "Start the tracker, look at each target on your display, and click the matching button while keeping your head still. "
        "Five samples work, nine points are better."
    )
    calibration_rows = [CALIBRATION_POINTS[:3], CALIBRATION_POINTS[3:6], CALIBRATION_POINTS[6:]]
    for row in calibration_rows:
        columns = st.columns(3)
        for column, (label, target) in zip(columns, row):
            with column:
                if st.button(label, use_container_width=True):
                    runtime.add_calibration_sample(*target)
                    st.rerun()
    st.caption(
        "Target order suggestion: top-left, top-center, top-right, mid-left, center, mid-right, bottom-left, bottom-center, bottom-right."
    )
    st.markdown("</section>", unsafe_allow_html=True)

    if state.last_error:
        st.error(state.last_error)

    st.markdown('<section class="card">', unsafe_allow_html=True)
    st.subheader("Live preview")
    status_left, status_mid, status_right = st.columns(3)
    status_left.metric("Concept", state.concept_label)
    status_mid.metric("Tracker FPS", f"{state.capture_fps:.1f}")
    status_right.metric("Mapping", "calibrated" if state.calibrated else "approximate")
    st.caption(state.debug_line)

    main_left, main_right = st.columns([1.7, 1.0], gap="large")
    with main_left:
        if state.screen_preview is not None:
            st.image(state.screen_preview, caption="Screen preview with gaze box", use_container_width=True)
        else:
            st.info("Start the tracker to see the live screen preview.")

    with main_right:
        if state.crop_preview is not None:
            st.image(state.crop_preview, caption="Current focus crop", use_container_width=True)
        else:
            st.info("The screenshot crop appears here once the gaze point is available.")

        if state.camera_preview is not None:
            st.image(state.camera_preview, caption="Webcam iris landmarks", use_container_width=True)
        else:
            st.info("The webcam preview appears here after the tracker starts.")
    st.markdown("</section>", unsafe_allow_html=True)

    with st.expander("Run notes", expanded=not state.running):
        st.markdown(
            f"""
            - App root: `{APP_ROOT}`
            - Recommended command: `streamlit run {APP_ROOT / "app.py"}`
            - Best accuracy comes after full 9-point calibration.
            - Uncalibrated mode still draws a box, but it is only an approximate gaze estimate.
            """
        )


if __name__ == "__main__":
    main()
