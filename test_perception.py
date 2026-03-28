from serial_reader import read_stream
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent

p = PerceptionAgent()
r = ReasoningAgent()

print(f"{'Z':>7}  {'Raw':>10}  {'Risk':>6}  State")
print("-"*45)

for v, z, m, s in read_stream():
    raw = p.classify(v, z)
    r.add(raw)
    state, risk = r.get_confirmed_state()
    print(f"{z:>7.2f}  {raw:>10}  {risk:>6.2f}  {state}")
