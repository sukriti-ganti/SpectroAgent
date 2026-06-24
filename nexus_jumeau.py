import numpy as np


class NexusJumeau:
    """First-order Markov chain over CLEAR / MOTION / ANOMALY.
    Transition counts initialised to ONE (Laplace smoothing) so the model is
    valid from the first frame. Predicts the most likely next state ~500 ms
    ahead, feeding the risk score and the resource agent."""

    def __init__(self):
        self.states = ["CLEAR", "MOTION", "ANOMALY"]
        self.counts = np.ones((3, 3))   # Laplace smoothing: never zero
        self.prev_state = None

    def state_index(self, state):
        mapping = {
            "CLEAR": 0, "HEIGHTENED": 0,
            "MOTION": 1, "PRE_ESCALATION": 1,
            "ANOMALY": 2, "EMERGENCY": 2, "CRITICAL": 2,
        }
        return mapping.get(state, 0)

    def update(self, confirmed_state):
        if self.prev_state is not None:
            i = self.state_index(self.prev_state)
            j = self.state_index(confirmed_state)
            self.counts[i][j] += 1
        self.prev_state = confirmed_state

    def predict_next(self, confirmed_state):
        i = self.state_index(confirmed_state)
        row = self.counts[i]
        probs = row / row.sum()
        nxt = int(np.argmax(probs))
        return self.states[nxt], round(float(probs[nxt]), 2)
