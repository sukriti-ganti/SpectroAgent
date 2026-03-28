from serial_reader import read_stream
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from nexus_jumeau import NexusJumeau
from resource_allocation_agent import ResourceAllocationAgent

perception = PerceptionAgent()
reasoning = ReasoningAgent()
nexus = NexusJumeau()
resource = ResourceAllocationAgent()

print("Step 4 — Nexus Jumeau + Resource Allocation\n")
print(f"{'Z':>6}  {'Raw':>10}  {'State':>15}  {'Risk':>5}  {'Next':>8}  {'Conf':>5}  {'Mode'}")
print("-"*75)

for variance, z, ema_mean, ema_std in read_stream():
    raw = perception.classify(variance, z)
    reasoning.add(raw)
    state, risk = reasoning.get_confirmed_state()
    nexus.update(state)
    predicted, confidence = nexus.predict_next(state)
    allocation = resource.allocate(risk)

    print(f"{z:>6.2f}  {raw:>10}  {state:>15}  {risk:>5.2f}  {predicted:>8}  {confidence:>5.2f}  {allocation['mode']}")
s