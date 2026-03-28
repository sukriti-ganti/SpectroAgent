from collections import deque

class ReasoningAgent:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
        self.weights = {"CLEAR": 0.0, "MOTION": 0.3, "ANOMALY": 1.0}

    def add(self, classification):
        self.window.append(classification)

    def compute_risk_score(self):
        if not self.window:
            return 0.0
        return sum(self.weights[w] for w in self.window) / len(self.window)

    def get_confirmed_state(self):
        if len(self.window) < 5:
            return "CLEAR", 0.0
        risk = self.compute_risk_score()
        if risk >= 0.70:
            return "CRITICAL", risk
        elif risk >= 0.50:
            return "EMERGENCY", risk
        elif risk >= 0.25:
            return "PRE_ESCALATION", risk
        elif risk >= 0.10:
            return "HEIGHTENED", risk
        return "CLEAR", risk