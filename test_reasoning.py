from serial_reader import read_stream
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent

perception = PerceptionAgent()
reasoning = ReasoningAgent()

print("🧠 Perception + Reasoning Agent running...\n")
print(f"{'Variance':>15}  {'Z':>6}  {'Raw':>10}  {'Risk':>6}  {'State'}")
print("-" * 60)

for variance, z, mean, std in read_stream():
    raw = perception.classify(variance, z)
    reasoning.add(raw)
    state, risk = reasoning.get_confirmed_state()

    symbol = {
        "CLEAR":          "🟢",
        "HEIGHTENED":     "🟡",
        "PRE_ESCALATION": "🟠",
        "EMERGENCY":      "🔴",
        "CRITICAL":       "🚨"
    }.get(state, "⚪")

    print(f"{variance:>15,.0f}  {z:>6.2f}  {raw:>10}  {risk:>6.2f}  {symbol} {state}")
