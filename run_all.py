#!/usr/bin/env python3
"""
One-click entry: train all single models + final ensemble.
Does not modify original model scripts (KNN.py, svm.py, etc.).

  python3 run_all.py          # full (includes slow mlp_1.4.py)
  python3 run_all.py --quick    # MLP single = sklearn ct_mlp export (fast smoke test)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
script = ROOT / "pipeline" / "run_all_submissions.py"
raise SystemExit(
    subprocess.run([sys.executable, str(script), *sys.argv[1:]], cwd=str(ROOT)).returncode
)
