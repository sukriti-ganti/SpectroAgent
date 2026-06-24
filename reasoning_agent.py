from collections import deque


class ReasoningAgent:
    """Window voting over recent frames. A single noisy frame cannot fire an
    alarm -- the signal must persist. This is the structural false-positive
    defense. Risk = weighted mean of the window; thresholds map risk to state."""

    def __init__(self, window_size=3):
        self.window = deque(maxlen=window_size)
        self.weights = {"CLEAR": 0.0, "MOTION": 0.3, "ANOMALY": 1.0}

    def add(self, classification):
        self.window.append(classification)

    def compute_risk_score(self):
        if not self.window:
            return 0.0
        return sum(self.weights[w] for w in self.window) / len(self.window)

    def get_confirmed_state(self):
        if len(self.window) < 3:
            return "CLEAR", 0.0
        risk = self.compute_risk_score()
        if risk >= 0.60:
            return "CRITICAL", risk
        elif risk >= 0.40:
            return "EMERGENCY", risk
        elif risk >= 0.20:
            return "PRE_ESCALATION", risk
        elif risk >= 0.08:
            return "HEIGHTENED", risk
        return "CLEAR", risk
