"""Run tests from the repo root so src/ imports and data/ relative paths resolve."""

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore", category=FutureWarning)
