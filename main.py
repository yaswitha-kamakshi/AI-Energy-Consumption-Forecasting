import subprocess
import sys

STEPS = [
    "src/preprocess.py",
    "src/features.py",
    "src/train.py",
    "src/evaluate.py",
    "src/forecast.py",
]

for step in STEPS:
    print(f"\n=== Running {step} ===")
    subprocess.run([sys.executable, step], check=True)

print("\nPipeline complete. See outputs/ and images/.")