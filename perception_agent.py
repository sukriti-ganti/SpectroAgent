from collections import deque


class PerceptionAgent:
    """Labels each frame CLEAR / MOTION / ANOMALY from the peak z-score
    over a short history. A linear decision rule on a normalised feature."""

    def __init__(self):
        self.history = deque(maxlen=5)

    def classify(self, feature, z_score):
        self.history.append(z_score)
        if len(self.history) < 2:
            return "CLEAR"
        peak = max(self.history)
        if peak > 2.0:
            return "ANOMALY"
        elif peak > 1.0:
            return "MOTION"
        return "CLEAR"
