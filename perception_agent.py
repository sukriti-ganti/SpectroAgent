from collections import deque

class PerceptionAgent:
    def __init__(self):
        self.history = deque(maxlen=5)

    def classify(self, variance, z_score):
        self.history.append(z_score)
        if len(self.history) < 2:
            return "CLEAR"
        peak = max(self.history)
        if peak > 2.0:
            return "ANOMALY"
        elif peak > 1.0:
            return "MOTION"
        return "CLEAR"