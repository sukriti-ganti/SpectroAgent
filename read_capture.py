import numpy as np

frames = np.load("captured_csi.npy", allow_pickle=True)
print(f"Total frames captured: {len(frames)}")

lengths = [len(f) for f in frames]
print(f"Values per frame: min={min(lengths)}, max={max(lengths)}, "
      f"most common={max(set(lengths), key=lengths.count)}")

print("\nFirst frame (first 20 values):")
print(list(frames[0])[:20])

print("\nLast frame (first 20 values):")
print(list(frames[-1])[:20])