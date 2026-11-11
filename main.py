"""Standard entry point: python main.py --input input.json --output output.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from agent import solve_problem


def _solve_item(item: Any, model: Optional[str]) -> dict:
    problem_id: Any = ""
    answer = ""
    try:
        if not isinstance(item, dict):
            raise ValueError("each input item must be an object with id and problem")
        problem_id = item.get("id", "")
        problem = item.get("problem", "")
        answer = solve_problem(problem if isinstance(problem, str) else str(problem), model=model)
    except Exception as exc:
        print(f"[agent] {exc}", file=sys.stderr, flush=True)
    return {"id": problem_id, "answer": str(answer)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local reasoning agent (Ollama)")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--model", default=None, help="Ollama model name (default: $OLLAMA_MODEL)")
    args = parser.parse_args(argv)

    try:
        data = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"[agent] could not read input: {exc}", file=sys.stderr, flush=True)
        data = {}

    # Spec: one object in, one object out. A list of objects is also accepted.
    if isinstance(data, list):
        result: Any = [_solve_item(item, args.model) for item in data]
    else:
        result = _solve_item(data, args.model)

    try:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        print(f"[agent] could not write output: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
