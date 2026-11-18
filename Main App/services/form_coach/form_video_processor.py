"""
Smart Form Coach – Video Processor
Uses the existing MediaPipe PoseLandmarker (VIDEO mode) and the
SmartSquatAnalyzer to annotate frames and expose metrics.
"""

import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading

from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from services.form_coach.squat_analyzer import SmartSquatAnalyzer
from services.config.workout_config import POSE_CONNECTIONS


# ── Colour palette ────────────────────────────────────────────────────────────
COL_GREEN   = (0,  220,  80)
COL_YELLOW  = (0,  200, 240)
COL_RED     = (0,   60, 220)
COL_WHITE   = (255, 255, 255)
COL_DARK    = (10,  10,  10)
COL_ACCENT  = (220,  80,  20)   # BGR orange/blue accent


class SmartFormVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor for the Smart Form Coach feature.
    Runs MediaPipe PoseLandmarker in VIDEO mode, delegates per-frame
    landmark analysis to SmartSquatAnalyzer, then draws rich overlays.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Build PoseLandmarker (same .task file as the main app)
        model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.65,
            min_pose_presence_confidence=0.65,
            min_tracking_confidence=0.65,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)
        self._frame_ts_ms = 0

        self._analyzer   = SmartSquatAnalyzer()
        self._latest     = {}
        self._active     = False   # True while workout is running (not paused)

    # ── External controls ─────────────────────────────────────────────────────
    def set_active(self, active: bool):
        with self._lock:
            self._active = active

    def reset_analyzer(self):
        self._analyzer.reset()
        with self._lock:
            self._latest = {}

    def get_latest_metrics(self):
        with self._lock:
            return self._latest.copy()

    # ── Frame receive callback ────────────────────────────────────────────────
    def recv(self, frame):
        # Flip horizontally for mirror view
        img = np.asarray(
            cv2.flip(frame.to_ndarray(format="bgr24"), 1),
            dtype=np.uint8
        )

        with self._lock:
            active = self._active

        if not active:
            # Paused / stopped: show greyed-out feed with banner
            self._draw_paused_banner(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Convert BGR→RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 30
        result = self._landmarker.detect_for_video(mp_img, self._frame_ts_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            metrics   = self._analyzer.process(landmarks)

            self._draw_skeleton(img, landmarks)
            self._draw_overlays(img, metrics)

            with self._lock:
                self._latest = metrics.copy()
        else:
            self._draw_no_pose(img)
            with self._lock:
                self._latest = {
                    "pose_detected": False,
                    "feedback": "⚠️ No pose detected — please face the camera.",
                    "status":   "NO POSE",
                }

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start, end in POSE_CONNECTIONS:
            p1, p2 = landmarks[start], landmarks[end]
            if p1.visibility > 0.65 and p2.visibility > 0.65:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    COL_GREEN, 5, cv2.LINE_AA
                )

        for lm in landmarks:
            if lm.visibility > 0.65:
                cv2.circle(
                    img,
                    (int(lm.x * w), int(lm.y * h)),
                    6, COL_ACCENT, -1
                )

    def _draw_overlays(self, img, metrics):
        h, w = img.shape[:2]
        status = metrics.get("status", "")

        # ── Status badge (top-left) ───────────────────────────────────────
        badge_col = COL_GREEN if "GOOD" in status or "REP" in status or "DEPTH" in status else \
                    COL_RED   if "ALERT" in status or "POOR" in status else \
                    COL_YELLOW
        cv2.rectangle(img, (10, 10), (w - 10, 54), COL_DARK, -1)
        cv2.rectangle(img, (10, 10), (w - 10, 54), badge_col, 2)
        cv2.putText(img, status, (22, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, badge_col, 2, cv2.LINE_AA)

        # ── Angle readouts (bottom) ───────────────────────────────────────
        knee_a = metrics.get("knee_angle", 0)
        back_a = metrics.get("back_angle", 0)
        y_base = h - 15

        cv2.rectangle(img, (0, h - 55), (w, h), COL_DARK, -1)
        cv2.putText(img, f"KNEE: {knee_a}deg",  (12, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"BACK: {back_a}deg",  (w // 2, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_WHITE, 1, cv2.LINE_AA)

        # ── Rep counter bubble (top-right) ────────────────────────────────
        reps = metrics.get("reps", 0)
        cv2.circle(img, (w - 50, 70), 42, COL_DARK, -1)
        cv2.circle(img, (w - 50, 70), 42, COL_GREEN, 2)
        label = str(reps)
        font_scale = 1.6 if reps < 10 else 1.1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(img, label, (w - 50 - tw // 2, 70 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COL_GREEN, 2, cv2.LINE_AA)

    def _draw_no_pose(self, img):
        h, w = img.shape[:2]
        cv2.rectangle(img, (10, 10), (w - 10, 54), COL_DARK, -1)
        cv2.rectangle(img, (10, 10), (w - 10, 54), COL_RED, 2)
        cv2.putText(img, "NO POSE DETECTED", (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_RED, 2, cv2.LINE_AA)
        cv2.putText(img, "Please face the camera", (22, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_YELLOW, 1, cv2.LINE_AA)

    def _draw_paused_banner(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        h, w = img.shape[:2]
        cv2.putText(img, "PAUSED", (w // 2 - 90, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COL_YELLOW, 3, cv2.LINE_AA)
