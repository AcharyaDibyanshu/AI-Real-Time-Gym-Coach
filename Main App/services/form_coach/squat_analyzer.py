"""
Smart Form Coach - Squat Analyzer
Provides enhanced squat analysis with form accuracy scoring,
rep quality tracking, and detailed posture feedback.
"""

import math
import threading
from core.base_exercise import BaseExercise


# ── Landmark indices (MediaPipe Pose) ────────────────────────────────────────
LEFT_HIP      = 23
LEFT_KNEE     = 25
LEFT_ANKLE    = 27
LEFT_SHOULDER = 11
LEFT_HEEL     = 29
LEFT_FOOT     = 31

RIGHT_HIP      = 24
RIGHT_KNEE     = 26
RIGHT_ANKLE    = 28
RIGHT_SHOULDER = 12
RIGHT_HEEL     = 30
RIGHT_FOOT     = 32


# ── Thresholds ────────────────────────────────────────────────────────────────
DOWN_THRESHOLD      = 105   # knee angle for "squat down" (< this = bottom)
UP_THRESHOLD        = 160   # knee angle for "standing" (> this = top)
PARALLEL_THRESHOLD  = 110   # knee angle ≤ this → good depth (thighs parallel)
BACK_LEAN_MIN       = 125   # back_angle below this → excessive forward lean
BACK_LEAN_GOOD_MIN  = 145   # back_angle above this → great upright posture
MIN_VISIBILITY      = 0.65


class SmartSquatAnalyzer:
    """
    State-machine-based squat analyzer that:
    - Counts reps using DOWN→UP transitions
    - Scores form accuracy per rep (0–100)
    - Generates contextual posture feedback
    - Tracks correct vs incorrect reps
    """

    # ── State machine states ──────────────────────────────────────────────────
    STATE_IDLE    = "idle"
    STATE_PREPARE = "prepare"
    STATE_DOWN    = "down"
    STATE_UP      = "up"

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._reps            = 0
            self._incorrect_reps  = 0
            self._state           = self.STATE_PREPARE
            self._rep_scores      = []          # score per completed rep
            self._current_rep_errors = []       # form errors during current descent
            self._form_accuracy   = 100.0
            self._feedback        = "Stand straight and prepare."
            self._status          = "PREPARE"
            self._knee_angle      = 0.0
            self._back_angle      = 0.0
            self._hip_angle       = 0.0
            self._pose_detected   = False
            self._reached_depth   = False       # did current rep reach proper depth?

    # ── Public read-only properties (thread-safe) ─────────────────────────────
    @property
    def reps(self):
        with self._lock:
            return self._reps

    @property
    def metrics(self):
        with self._lock:
            return {
                "reps":           self._reps,
                "incorrect_reps": self._incorrect_reps,
                "form_accuracy":  round(self._form_accuracy, 1),
                "feedback":       self._feedback,
                "status":         self._status,
                "knee_angle":     int(self._knee_angle),
                "back_angle":     int(self._back_angle),
                "hip_angle":      int(self._hip_angle),
                "pose_detected":  self._pose_detected,
                "depth_ok":       self._reached_depth,
            }

    # ── Core analysis (called per frame) ─────────────────────────────────────
    def process(self, landmarks):
        """
        Analyse a single frame of landmarks.
        Returns the latest metrics dict.
        """
        # pick the more visible side
        l_vis = landmarks[LEFT_KNEE].visibility
        r_vis = landmarks[RIGHT_KNEE].visibility

        if l_vis >= r_vis:
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_SHOULDER)
        else:
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_SHOULDER)

        # check visibility
        key_visible = all(
            landmarks[i].visibility >= MIN_VISIBILITY
            for i in (hip_idx, knee_idx, ankle_idx, shoulder_idx)
        )

        with self._lock:
            if not key_visible:
                self._pose_detected = False
                self._feedback = "⚠️ Please step back so your full body is visible."
                self._status = "NO POSE"
                return self._build_metrics()

            self._pose_detected = True

            # ── Calculate angles ──────────────────────────────────────────
            knee_angle = _angle(
                _pt(landmarks, hip_idx),
                _pt(landmarks, knee_idx),
                _pt(landmarks, ankle_idx)
            )

            back_angle = _angle(
                _pt(landmarks, shoulder_idx),
                _pt(landmarks, hip_idx),
                _pt(landmarks, knee_idx)
            )

            hip_angle = _angle(
                _pt(landmarks, shoulder_idx),
                _pt(landmarks, hip_idx),
                _pt(landmarks, ankle_idx)
            )

            self._knee_angle = knee_angle
            self._back_angle = back_angle
            self._hip_angle  = hip_angle

            # ── State machine ─────────────────────────────────────────────
            self._run_state_machine(knee_angle, back_angle)

            return self._build_metrics()

    def _run_state_machine(self, knee_angle, back_angle):
        """Progress the rep state machine and collect per-rep form errors."""
        state = self._state

        # ── PREPARE → entering squat ───────────────────────────────────────
        if state == self.STATE_PREPARE:
            if knee_angle < DOWN_THRESHOLD:
                # started going down
                self._state = self.STATE_DOWN
                self._current_rep_errors = []
                self._reached_depth = False
                self._feedback = "Going down — keep your back straight!"
                self._status = "DOWN"
            else:
                self._feedback, self._status = self._prepare_feedback(back_angle, knee_angle)

        # ── DOWN phase ─────────────────────────────────────────────────────
        elif state == self.STATE_DOWN:
            # check if reached good depth
            if knee_angle <= PARALLEL_THRESHOLD:
                self._reached_depth = True

            # collect errors during descent
            err = self._form_error(back_angle, knee_angle, in_down_phase=True)
            if err and err not in self._current_rep_errors:
                self._current_rep_errors.append(err)

            # generate live feedback
            self._feedback, self._status = self._down_feedback(
                back_angle, knee_angle, self._reached_depth)

            # transition to UP when knee extends again
            if knee_angle >= UP_THRESHOLD:
                self._complete_rep()
                self._state = self.STATE_PREPARE

        # ── UP phase (momentary, used only as a counted-rep confirmation) ──
        elif state == self.STATE_UP:
            self._state = self.STATE_PREPARE
            self._feedback = "✅ Rep completed! Prepare for the next one."
            self._status = "UP"

    def _complete_rep(self):
        """Score and record the completed rep."""
        # Base score = 100; deduct for each form error
        score = 100
        for err in self._current_rep_errors:
            score -= 20  # -20 per distinct error type

        if not self._reached_depth:
            score -= 25  # penalty for not reaching parallel

        score = max(0, score)
        self._rep_scores.append(score)

        if score >= 70:
            self._reps += 1
            self._feedback = f"✅ Rep {self._reps} complete! Great work! 💪"
            self._status = "REP DONE"
        else:
            self._incorrect_reps += 1
            self._reps += 1          # still count the rep
            self._feedback = f"⚠️ Rep {self._reps} counted — fix your form next time."
            self._status = "POOR FORM REP"

        # Recalculate rolling form accuracy
        if self._rep_scores:
            self._form_accuracy = sum(self._rep_scores) / len(self._rep_scores)
        else:
            self._form_accuracy = 100.0

        self._current_rep_errors = []
        self._reached_depth = False

    # ── Feedback helpers ──────────────────────────────────────────────────────
    def _prepare_feedback(self, back_angle, knee_angle):
        if knee_angle > 170:
            return "Stand straight and prepare. When ready, begin squatting.", "PREPARE"
        if back_angle < BACK_LEAN_MIN:
            return "Keep your back straight before starting.", "BACK ALERT"
        return "Good stance! Begin your squat.", "READY"

    def _down_feedback(self, back_angle, knee_angle, reached_depth):
        if back_angle < BACK_LEAN_MIN:
            return "⚠️ Keep your back straight!", "BACK ALERT"
        if knee_angle > 130 and not reached_depth:
            return "Bend your knees more — go deeper!", "TOO HIGH"
        if reached_depth:
            return "🔥 Good depth! Drive back up!", "GOOD DEPTH"
        return "Going down — keep your chest up.", "DESCENDING"

    def _form_error(self, back_angle, knee_angle, in_down_phase):
        if back_angle < BACK_LEAN_MIN:
            return "forward_lean"
        return None

    def _build_metrics(self):
        return {
            "reps":           self._reps,
            "incorrect_reps": self._incorrect_reps,
            "form_accuracy":  round(self._form_accuracy, 1),
            "feedback":       self._feedback,
            "status":         self._status,
            "knee_angle":     int(self._knee_angle),
            "back_angle":     int(self._back_angle),
            "hip_angle":      int(self._hip_angle),
            "pose_detected":  self._pose_detected,
            "depth_ok":       self._reached_depth,
        }


# ── Utility functions ─────────────────────────────────────────────────────────
def _pt(landmarks, idx):
    p = landmarks[idx]
    return (p.x, p.y)


def _angle(a, b, c):
    """Calculate the angle (degrees) at vertex b, formed by a-b-c."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag_a = math.sqrt(ax**2 + ay**2)
    mag_c = math.sqrt(cx**2 + cy**2)
    if mag_a * mag_c == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_a))
