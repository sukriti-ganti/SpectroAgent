class ResourceAllocationAgent:
    def allocate(self, risk_score):
        if risk_score < 0.10:
            return {"mode": "LOW POWER", "interval": 0.5}
        elif risk_score < 0.30:
            return {"mode": "HEIGHTENED", "interval": 0.1}
        elif risk_score < 0.60:
            return {"mode": "PRE-ESCALATION", "interval": 0.075}
        elif risk_score < 0.85:
            return {"mode": "EMERGENCY", "interval": 0.05}
        else:
            return {"mode": "CRITICAL", "interval": 0.025}