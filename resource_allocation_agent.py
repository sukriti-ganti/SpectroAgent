class ResourceAllocationAgent:
    """The ISAC knob. Maps the risk score to a sensing rate / CIC decimation
    factor. Low risk -> decimate hard, free the radio for communication.
    High risk -> full rate, all resources on sensing. R is computed live."""

    def allocate(self, risk_score):
        if risk_score < 0.10:
            return {"mode": "LOW POWER", "interval": 0.5, "R": 8}
        elif risk_score < 0.30:
            return {"mode": "HEIGHTENED", "interval": 0.1, "R": 4}
        elif risk_score < 0.60:
            return {"mode": "PRE-ESCALATION", "interval": 0.075, "R": 2}
        elif risk_score < 0.85:
            return {"mode": "EMERGENCY", "interval": 0.05, "R": 1}
        else:
            return {"mode": "CRITICAL", "interval": 0.025, "R": 1}
