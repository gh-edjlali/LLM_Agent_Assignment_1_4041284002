"""Standard entry point: python main.py --input input.json --output output.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import solve_problem


def load_input(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("input JSON must be an object with id and problem")
    return data


def write_output(path: Path, problem_id: object, answer: str) -> None:
    payload = {"id": problem_id, "answer": answer}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local reasoning agent")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--model", default=None, help="Ollama model name")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    problem_id: object = ""
    try:
        data = load_input(Path(args.input))
        problem_id = data.get("id", "")
        problem = data.get("problem", "")
        if not isinstance(problem, str):
            problem = str(problem)
        answer = solve_problem(problem, model=args.model)
    except Exception as exc:
        print(f"[agent] {exc}", file=sys.stderr, flush=True)
        answer = ""

    try:
        write_output(output_path, problem_id, answer)
    except Exception as exc:
        print(f"[agent] could not write output: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
