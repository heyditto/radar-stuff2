"""Fill held-out blockage gaps using the tilts above and below, when present."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from radar.holdout_gaps import main

if __name__ == "__main__":
    main()
