import numpy as np
from collections import deque

class PerceptionAgent:
    def __init__(self):
        self.history = deque(maxlen=10)

    def classify(self, variance, z_score):
        self.history.append(z_score)
        if len(self.history) < 3:
            return "CLEAR"
        recent_max = max(self.history)
        if recent_max > 2.5:
            return "ANOMALY"
        elif recent_max > 1.2:
            return "MOTION"
        return "CLEAR"