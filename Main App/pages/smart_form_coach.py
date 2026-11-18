"""
Smart Form Coach Page
Rendered by main.py when the user selects "Smart Form Coach" in the sidebar.
"""

import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from services.form_coach.form_video_processor import SmartFormVideoProcessor
from services.ui.style_loader import inject_webrtc_styles


# ── Exercise catalogue (expandable later) ─────────────────────────────────────
SUPPORTED_EXERCISES = {
    "Squats": {
        "icon":        "🏋️",
        "description": "Full body compound movement. Focus on knee tracking, back angle, and squat depth.",
        "tips": [
            "Keep your feet shoulder-width apart.",
            "Track your knees over your toes.",
            "Keep your chest up and back straight.",
            "Break parallel for optimal depth.",
        ],
    }
}


def _init_sfc_state():
    """Ensure all Smart Form Coach session keys have defaults."""
    defaults = {
        "sfc_active":        False,     # workout running (not paused)
        "sfc_paused":        False,
        "sfc_stopped":       True,
        "sfc_start_time":    None,
        "sfc_pause_elapsed": 0.0,       # seconds already counted before pause
        "sfc_exercise":      "Squats",
        "sfc_summary":       None,      # dict filled when workout stops
        # live metrics cache
        "sfc_reps":          0,
        "sfc_incorrect":     0,
        "sfc_accuracy":      100.0,
        "sfc_feedback":      "Press Start Workout to begin.",
        "sfc_status":        "IDLE",
        "sfc_knee_angle":    0,
        "sfc_back_angle":    0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _elapsed_seconds() -> float:
    """Return total workout seconds (accounting for pauses)."""
    if st.session_state.sfc_start_time is None:
        return 0.0
    if st.session_state.sfc_paused or st.session_state.sfc_stopped:
        return st.session_state.sfc_pause_elapsed
    return st.session_state.sfc_pause_elapsed + (time.time() - st.session_state.sfc_start_time)


def _fmt_duration(secs: float) -> str:
    secs = int(secs)
    return f"{secs // 60:02d}:{secs % 60:02d}"


# ── Main render function ──────────────────────────────────────────────────────
def render_smart_form_coach():
    _init_sfc_state()

    st.title("🤖 Smart Form Coach")
    st.markdown("#### Real-time pose analysis · Rep counting · Form accuracy")
    st.divider()

    # ── Exercise selector ─────────────────────────────────────────────────────
    if st.session_state.sfc_stopped:
        _render_exercise_selector()

    # ── Webcam feed + dashboard ────────────────────────────────────────────────
    _render_webcam_section()

    # ── Summary card ─────────────────────────────────────────────────────────
    if st.session_state.sfc_summary:
        _render_summary()


# ── Sub-sections ──────────────────────────────────────────────────────────────

def _render_exercise_selector():
    ex_name = st.selectbox(
        "Choose Exercise",
        options=list(SUPPORTED_EXERCISES.keys()),
        key="sfc_exercise",
        help="More exercises coming soon.",
    )
    info = SUPPORTED_EXERCISES[ex_name]
    with st.expander(f"{info['icon']} About this exercise", expanded=False):
        st.markdown(info["description"])
        st.markdown("**Form Tips:**")
        for tip in info["tips"]:
            st.markdown(f"- {tip}")
    st.markdown("")


def _render_webcam_section():
    # ── Controls row ──────────────────────────────────────────────────────────
    col_start, col_pause, col_stop = st.columns(3)

    with col_start:
        start_disabled = st.session_state.sfc_active
        if st.button("▶ Start Workout", disabled=start_disabled,
                     key="sfc_btn_start", use_container_width=True):
            _do_start()
            st.rerun()

    with col_pause:
        pause_disabled = st.session_state.sfc_stopped
        pause_label    = "⏸ Pause" if st.session_state.sfc_active else "▶ Resume"
        if st.button(pause_label, disabled=pause_disabled,
                     key="sfc_btn_pause", use_container_width=True):
            _do_pause_resume()
            st.rerun()

    with col_stop:
        stop_disabled = st.session_state.sfc_stopped
        if st.button("⏹ Stop Workout", disabled=stop_disabled,
                     key="sfc_btn_stop", use_container_width=True):
            _do_stop()
            st.rerun()

    st.markdown("")

    # ── WebRTC stream ─────────────────────────────────────────────────────────
    if st.session_state.sfc_stopped and st.session_state.sfc_summary is None:
        # Pre-workout placeholder
        st.markdown(
            """
            <div style="
                border: 8px dashed #333;
                padding: 56px 32px;
                text-align: center;
                color: #666;
                margin-bottom: 24px;
            ">
                <h3 style="color:#aaa; margin:0 0 8px;">📷 Webcam Preview</h3>
                <p>Select an exercise and press <strong>Start Workout</strong><br>
                to activate the camera and begin pose analysis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Build the processor factory (must be a callable returning the same instance
    # across re-renders so we can read metrics from it).
    if "sfc_processor_key" not in st.session_state:
        st.session_state.sfc_processor_key = "sfc-stream-v1"

    ctx = webrtc_streamer(
        key=st.session_state.sfc_processor_key,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SmartFormVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    inject_webrtc_styles()

    # Sync processor active state
    if ctx and ctx.video_processor:
        ctx.video_processor.set_active(st.session_state.sfc_active)

    # ── Live metrics sync ──────────────────────────────────────────────────────
    if ctx and ctx.state.playing and st.session_state.sfc_active:
        _sync_metrics(ctx)
        time.sleep(0.25)
        st.rerun()

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if not st.session_state.sfc_stopped:
        _render_live_dashboard()


def _sync_metrics(ctx):
    """Pull latest metrics from the video processor into session state."""
    if not ctx or not ctx.video_processor:
        return
    m = ctx.video_processor.get_latest_metrics()
    if not m:
        return

    st.session_state.sfc_reps       = m.get("reps",          st.session_state.sfc_reps)
    st.session_state.sfc_incorrect   = m.get("incorrect_reps", st.session_state.sfc_incorrect)
    st.session_state.sfc_accuracy    = m.get("form_accuracy",  st.session_state.sfc_accuracy)
    st.session_state.sfc_feedback    = m.get("feedback",        st.session_state.sfc_feedback)
    st.session_state.sfc_status      = m.get("status",          st.session_state.sfc_status)
    st.session_state.sfc_knee_angle  = m.get("knee_angle",      st.session_state.sfc_knee_angle)
    st.session_state.sfc_back_angle  = m.get("back_angle",      st.session_state.sfc_back_angle)


def _render_live_dashboard():
    st.markdown("### 📊 Live Dashboard")

    elapsed = _elapsed_seconds()

    # ── Top metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔄 Reps",          st.session_state.sfc_reps)
    c2.metric("🎯 Form Accuracy", f"{st.session_state.sfc_accuracy:.0f}%")
    c3.metric("⏱ Duration",      _fmt_duration(elapsed))
    c4.metric("❌ Poor Reps",     st.session_state.sfc_incorrect)

    st.markdown("")

    # ── Angle readouts ────────────────────────────────────────────────────────
    a1, a2 = st.columns(2)
    a1.metric("🦵 Knee Angle", f"{st.session_state.sfc_knee_angle}°")
    a2.metric("🧍 Back Angle", f"{st.session_state.sfc_back_angle}°")

    st.markdown("")

    # ── Feedback banner ───────────────────────────────────────────────────────
    feedback = st.session_state.sfc_feedback
    status   = st.session_state.sfc_status

    if "ALERT" in status or "POOR" in status or "NO POSE" in status:
        st.error(f"**Posture Feedback:** {feedback}")
    elif "GOOD" in status or "REP" in status or "DEPTH" in status:
        st.success(f"**Posture Feedback:** {feedback}")
    else:
        st.info(f"**Posture Feedback:** {feedback}")

    # ── Form accuracy bar ─────────────────────────────────────────────────────
    accuracy = st.session_state.sfc_accuracy / 100.0
    st.markdown(
        f"""
        <div style="margin: 8px 0 4px;">
            <span style="font-size:0.85rem; color:#aaa; letter-spacing:0.05em;">
                FORM ACCURACY
            </span>
        </div>
        <div style="background:#1a1f2e; border-radius:0; height:14px; width:100%; border:1px solid #333;">
            <div style="
                background: {'#00dc50' if accuracy > 0.75 else '#f0a800' if accuracy > 0.5 else '#e03030'};
                height: 100%;
                width: {accuracy * 100:.1f}%;
                transition: width 0.4s ease;
            "></div>
        </div>
        <div style="color:#888; font-size:0.78rem; margin-top:4px;">
            {accuracy * 100:.1f}% accuracy across {st.session_state.sfc_reps} rep(s)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Status pill ───────────────────────────────────────────────────────────
    pill_col = ("#00dc50" if "GOOD" in status or "DEPTH" in status or "DONE" in status
                else "#e03030" if "ALERT" in status or "POOR" in status
                else "#f0a800")
    st.markdown(
        f"""<div style="
            display:inline-block; margin-top:12px;
            background:{pill_col}22;
            border:1px solid {pill_col};
            padding: 4px 16px;
            font-size: 0.88rem;
            letter-spacing: 0.08em;
            color: {pill_col};
        ">STATUS: {status}</div>""",
        unsafe_allow_html=True,
    )


def _render_summary():
    s = st.session_state.sfc_summary
    st.divider()
    st.markdown("## 🏁 Workout Summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Reps",    s["reps"])
    s2.metric("Form Accuracy", f"{s['accuracy']:.0f}%")
    s3.metric("Duration",      _fmt_duration(s["duration"]))
    s4.metric("Poor Reps",     s["incorrect_reps"])

    if s["reps"] > 0:
        pct_good = (s["reps"] - s["incorrect_reps"]) / s["reps"] * 100
        if pct_good >= 80:
            st.success(f"🏆 Excellent session! {pct_good:.0f}% of your reps had good form.")
        elif pct_good >= 50:
            st.warning(f"💪 Good effort! {pct_good:.0f}% of your reps had good form. Keep practising!")
        else:
            st.error(f"🔧 {pct_good:.0f}% good-form reps — focus on technique before adding reps.")
    else:
        st.info("No reps completed in this session.")

    if st.button("🔄 New Session", key="sfc_new_session"):
        # Clear summary and reset state for a fresh session
        st.session_state.sfc_summary = None
        for k in ["sfc_active","sfc_paused","sfc_stopped","sfc_start_time",
                  "sfc_pause_elapsed","sfc_reps","sfc_incorrect",
                  "sfc_accuracy","sfc_feedback","sfc_status",
                  "sfc_knee_angle","sfc_back_angle"]:
            del st.session_state[k]
        st.rerun()


# ── State-transition helpers ──────────────────────────────────────────────────
def _do_start():
    st.session_state.sfc_active       = True
    st.session_state.sfc_paused       = False
    st.session_state.sfc_stopped      = False
    st.session_state.sfc_start_time   = time.time()
    st.session_state.sfc_pause_elapsed = 0.0
    st.session_state.sfc_summary      = None
    # reset live metrics
    st.session_state.sfc_reps         = 0
    st.session_state.sfc_incorrect    = 0
    st.session_state.sfc_accuracy     = 100.0
    st.session_state.sfc_feedback     = "Workout started! Stand straight and prepare."
    st.session_state.sfc_status       = "PREPARE"
    # Force a new webrtc key so the processor is fresh
    st.session_state.sfc_processor_key = f"sfc-stream-{int(time.time())}"


def _do_pause_resume():
    if st.session_state.sfc_active:
        # Pause
        elapsed = time.time() - st.session_state.sfc_start_time
        st.session_state.sfc_pause_elapsed += elapsed
        st.session_state.sfc_start_time = None
        st.session_state.sfc_active  = False
        st.session_state.sfc_paused  = True
        st.session_state.sfc_feedback = "⏸ Workout paused."
    else:
        # Resume
        st.session_state.sfc_start_time = time.time()
        st.session_state.sfc_active  = True
        st.session_state.sfc_paused  = False
        st.session_state.sfc_feedback = "▶ Workout resumed — let's go!"


def _do_stop():
    # Finalise elapsed time
    if st.session_state.sfc_active and st.session_state.sfc_start_time:
        elapsed = time.time() - st.session_state.sfc_start_time
        st.session_state.sfc_pause_elapsed += elapsed

    duration = st.session_state.sfc_pause_elapsed

    st.session_state.sfc_summary = {
        "reps":         st.session_state.sfc_reps,
        "incorrect_reps": st.session_state.sfc_incorrect,
        "accuracy":     st.session_state.sfc_accuracy,
        "duration":     duration,
        "exercise":     st.session_state.sfc_exercise,
    }

    st.session_state.sfc_active  = False
    st.session_state.sfc_paused  = False
    st.session_state.sfc_stopped = True
    st.session_state.sfc_start_time    = None
    st.session_state.sfc_pause_elapsed = 0.0
