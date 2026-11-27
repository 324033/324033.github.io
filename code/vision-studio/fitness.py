"""
Fitness coach logic: arm rep counting based on MediaPipe Pose landmarks.

One rep = arm goes from straight (extended) -> bent -> straight again.
Left and right arms are counted independently. "Left" / "right" refer to the
person's own arms (MediaPipe pose landmark convention), not screen sides.
"""
import math

import cv2

# MediaPipe Pose landmark indices (shoulder, elbow, wrist)
ARM_LANDMARKS = {
    'left': (11, 13, 15),
    'right': (12, 14, 16),
}

DEFAULT_EXTENDED_ANGLE = 150.0   # elbow angle >= this counts as "straight"
DEFAULT_BENT_ANGLE = 60.0        # elbow angle <= this counts as "bent"
MIN_VISIBILITY = 0.5

# BGR colors
ARM_COLORS = {'left': (255, 220, 0), 'right': (0, 140, 255)}
SKELETON_COLOR = (200, 200, 200)


def elbow_angle(shoulder, elbow, wrist):
    """Angle at the elbow in degrees (0 = fully folded, 180 = fully straight).
    Points are (x, y) in pixel space so the frame aspect ratio does not distort it."""
    ax, ay = shoulder[0] - elbow[0], shoulder[1] - elbow[1]
    bx, by = wrist[0] - elbow[0], wrist[1] - elbow[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return None
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))


class ArmRepCounter:
    """State machine for one arm with hysteresis between the two thresholds.

    States: 'waiting' (no straight arm seen yet) -> 'extended' <-> 'bent'.
    A rep is counted on the transition bent -> extended.
    """

    def __init__(self, extended_angle=DEFAULT_EXTENDED_ANGLE, bent_angle=DEFAULT_BENT_ANGLE,
                 smoothing=0.5):
        self.extended_angle = extended_angle
        self.bent_angle = bent_angle
        self.smoothing = smoothing
        self.reset()

    def reset(self):
        self.reps = 0
        self.state = 'waiting'
        self.angle = None
        self.visible = False
        self.feedback = 'Straighten your arm to start'
        self._min_since_extended = 180.0
        self._max_since_bent = 0.0

    def update(self, raw_angle):
        """Feed one frame's elbow angle (None if the arm is not visible).
        Returns True if a rep was completed on this frame."""
        if raw_angle is None:
            self.visible = False
            return False
        self.visible = True

        if self.angle is None:
            self.angle = raw_angle
        else:
            self.angle = self.smoothing * self.angle + (1 - self.smoothing) * raw_angle
        a = self.angle

        if self.state == 'waiting':
            if a >= self.extended_angle:
                self._enter_extended()
                self.feedback = 'Ready - bend your arm'
            return False

        if self.state == 'extended':
            self._min_since_extended = min(self._min_since_extended, a)
            if a <= self.bent_angle:
                self.state = 'bent'
                self._max_since_bent = a
                self.feedback = 'Good - now straighten fully'
            elif a >= self.extended_angle and self._min_since_extended < self.extended_angle - 30:
                # came back up without reaching the bent threshold -> partial rep
                self.feedback = f'Not counted - bend further (below {int(self.bent_angle)}°)'
                self._min_since_extended = a
            return False

        # state == 'bent'
        self._max_since_bent = max(self._max_since_bent, a)
        if a >= self.extended_angle:
            self.reps += 1
            self._enter_extended()
            self.feedback = f'Rep {self.reps}!'
            return True
        if a <= self.bent_angle and self._max_since_bent > self.bent_angle + 40:
            # went partly up and back down again without straightening
            self.feedback = f'Straighten fully (above {int(self.extended_angle)}°) to count'
            self._max_since_bent = a
        return False

    def _enter_extended(self):
        self.state = 'extended'
        self._min_since_extended = self.angle

    def to_dict(self):
        return {
            'reps': self.reps,
            'state': self.state,
            'angle': None if self.angle is None else round(self.angle, 1),
            'visible': self.visible,
            'feedback': self.feedback,
        }


class ArmRepTracker:
    """Counts reps for both arms from pose landmarks."""

    def __init__(self, extended_angle=DEFAULT_EXTENDED_ANGLE, bent_angle=DEFAULT_BENT_ANGLE):
        self.arms = {side: ArmRepCounter(extended_angle, bent_angle) for side in ARM_LANDMARKS}

    def update(self, pose_landmarks, width, height):
        """pose_landmarks: list of landmarks for ONE person (or None)."""
        for side, (s, e, w) in ARM_LANDMARKS.items():
            angle = None
            if pose_landmarks:
                pts = [pose_landmarks[i] for i in (s, e, w)]
                vis = min(getattr(p, 'visibility', 1.0) or 0.0 for p in pts)
                if vis >= MIN_VISIBILITY:
                    angle = elbow_angle(*[(p.x * width, p.y * height) for p in pts])
            self.arms[side].update(angle)

    @property
    def counts(self):
        return {side: c.reps for side, c in self.arms.items()}


def annotate_pose_reps(image_bgr, pose_landmarks, tracker):
    """Draws the pose skeleton, highlighted arms with elbow angles and a rep HUD."""
    from mediapipe.tasks.python.vision import PoseLandmarksConnections

    canvas = image_bgr.copy()
    h, w, _ = canvas.shape
    scale = max(0.6, min(1.6, w / 960.0))

    if pose_landmarks:
        pts = [(int(round(lm.x * w)), int(round(lm.y * h))) for lm in pose_landmarks]
        vis = [getattr(lm, 'visibility', 1.0) or 0.0 for lm in pose_landmarks]

        for conn in PoseLandmarksConnections.POSE_LANDMARKS:
            if conn.start <= 10 or conn.end <= 10:
                continue  # skip face points for a cleaner coach view
            if vis[conn.start] < MIN_VISIBILITY or vis[conn.end] < MIN_VISIBILITY:
                continue
            cv2.line(canvas, pts[conn.start], pts[conn.end], SKELETON_COLOR, max(1, int(2 * scale)), cv2.LINE_AA)

        for side, (s, e, wr) in ARM_LANDMARKS.items():
            counter = tracker.arms[side]
            if not counter.visible:
                continue
            color = ARM_COLORS[side]
            thick = max(2, int(5 * scale))
            cv2.line(canvas, pts[s], pts[e], color, thick, cv2.LINE_AA)
            cv2.line(canvas, pts[e], pts[wr], color, thick, cv2.LINE_AA)
            for i in (s, e, wr):
                cv2.circle(canvas, pts[i], max(3, int(6 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pts[i], max(3, int(6 * scale)), color, 2, cv2.LINE_AA)
            if counter.angle is not None:
                txt = f"{int(round(counter.angle))} deg"  # OpenCV fonts have no degree sign
                org = (pts[e][0] + 10, pts[e][1] - 10)
                cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_DUPLEX, 0.6 * scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_DUPLEX, 0.6 * scale, color, 1, cv2.LINE_AA)

    # HUD panel bottom-left
    panel_w, panel_h = int(300 * scale), int(92 * scale)
    x0, y0 = int(12 * scale), h - panel_h - int(12 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    for row, side in enumerate(('left', 'right')):
        c = tracker.arms[side]
        y = y0 + int((36 + row * 40) * scale)
        label = f"{side.upper()} ARM: {c.reps}"
        state = c.state if c.visible else 'not visible'
        cv2.putText(canvas, label, (x0 + int(12 * scale), y), cv2.FONT_HERSHEY_DUPLEX, 0.8 * scale,
                    ARM_COLORS[side], max(1, int(2 * scale)), cv2.LINE_AA)
        cv2.putText(canvas, state, (x0 + int(190 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return canvas
