"""Run main.py on every file in examples/public and print the answers + time.

Usage: python run_public.py [--model qwen3.6:27b]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).parent
out_dir = here / "examples" / "outputs"
out_dir.mkdir(exist_ok=True)
for path in sorted((here / "examples" / "public").glob("*.json")):
    out = out_dir / path.name
    start = time.time()
    subprocess.run([sys.executable, str(here / "main.py"), "--input", str(path), "--output", str(out), *sys.argv[1:]])
    answer = json.loads(out.read_text(encoding="utf-8"))["answer"]
    print(f"{path.stem}: {answer!r}  ({time.time() - start:.0f}s)", flush=True)
