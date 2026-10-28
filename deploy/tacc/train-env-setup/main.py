"""Install LoRA training libraries into USERDIR/pylib and check they import (CPU job).

Only pure-Python wheels are installed (--no-deps), so the course torch/transformers stay
untouched. pip comes from the user's own Miniconda; the course vLLM env has no pip.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

USERDIR = Path(os.environ["TACC_USERDIR"])
PYLIB = USERDIR / "pylib"
PACKAGES = ["peft", "accelerate", "flash-linear-attention", "fla-core"]
env = {**os.environ, "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
       "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
report = {"job": os.environ.get("SLURM_JOB_ID"), "pylib": str(PYLIB), "installs": {}}

pip_python = os.environ.get("CONDA_PYTHON_EXE", "/mnt/home/markzhang/.Miniconda3/bin/python")
report["pip_python"] = pip_python
PYLIB.mkdir(parents=True, exist_ok=True)
for pkg in PACKAGES:
    # Miniconda's pip runs on Python 3.8, so it must be told to ignore Requires-Python; the wheels
    # are pure Python and are used later by the course Python 3.12.
    cmd = [pip_python, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
           "--ignore-requires-python",
           "--python-version", "3.12", "--implementation", "cp", "--target", str(PYLIB),
           "--upgrade", pkg]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    report["installs"][pkg] = {"returncode": out.returncode,
                               "tail": (out.stdout + out.stderr)[-500:]}
    print(pkg, "rc", out.returncode, flush=True)

# Import check in the course interpreter (this one), with pylib on the path.
sys.path.insert(0, str(PYLIB))
checks = {}
for mod in ["peft", "accelerate", "fla", "torch", "transformers", "compressed_tensors"]:
    try:
        m = __import__(mod)
        checks[mod] = getattr(m, "__version__", "ok")
    except Exception as exc:
        checks[mod] = f"IMPORT FAILED: {exc!r}"[:300]
report["imports"] = checks
report["installed_dirs"] = sorted(p.name for p in PYLIB.iterdir())[:60]
out = USERDIR / "tacc-agentforge-results"
out.mkdir(parents=True, exist_ok=True)
(out / f"train-env-setup-{report['job']}.json").write_text(json.dumps(report, indent=2))
print("REPORT", json.dumps(report, indent=2), flush=True)
