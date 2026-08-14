import subprocess, sys

venv_python = sys.executable  # use current Python interpreter
packages = ["transformers", "huggingface_hub", "timm", "tqdm"]

for pkg in packages:
    print(f"Installing {pkg}...")
    result = subprocess.run([venv_python, "-m", "pip", "install", pkg, "-q"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  OK: {pkg}")
    else:
        print(f"  FAIL: {pkg}")
        print(f"  {result.stderr[:200]}")

# Verify
print("\nVerification:")
result = subprocess.run([venv_python, "-c", "import torch; print('torch OK'); import transformers; print('transformers', transformers.__version__); import timm; print('timm OK'); import huggingface_hub; print('hf_hub OK')"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:300])
