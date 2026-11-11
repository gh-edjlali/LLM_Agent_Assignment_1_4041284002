"""Local checks that do not call the model.  Run: python test_unit.py"""

import json
from pathlib import Path

from agent import answer_matches_contract, canonicalize, format_hint, normalize_digits, parse_solver
from sandbox import SandboxError, run_python

PUBLIC = Path(__file__).parent / "examples" / "public"


def _problem(pid: str) -> str:
    return normalize_digits(json.loads((PUBLIC / f"{pid}.json").read_text(encoding="utf-8"))["problem"])


def _raises(code: str) -> bool:
    try:
        run_python(code)
    except SandboxError:
        return True
    return False


def test_digits() -> None:
    assert normalize_digits("۷۲ و ٢٤") == "72 و 24"


def test_parse_json() -> None:
    answer, code = parse_solver('{"answer": "8", "python": "print(8)\\n"}')
    assert answer == "8" and "print(8)" in code
    answer, code = parse_solver('{"answer": 8, "python": "print(8)"}')
    assert answer == "8"


def test_sandbox() -> None:
    assert run_python("print(40 - (24 + 18 - 10))\n") == "8"
    assert run_python("import re\nprint(re.compile(r'a').sub('b', 'a'))") == "b"
    assert _raises("import subprocess\nprint(1)\n")
    assert _raises("print(open('/etc/passwd').read())")
    assert _raises("print(1)\nprint(2)\n")
    assert _raises("while True: pass")


def test_contracts() -> None:
    p1 = _problem("public_001")
    assert answer_matches_contract(p1, "Nima=30, Sara=27, Reza=15")
    assert not answer_matches_contract(p1, "30, 27, 15")
    p2 = _problem("public_002")
    assert answer_matches_contract(p2, "C") and not answer_matches_contract(p2, "CA")
    p5 = _problem("public_005")
    assert answer_matches_contract(p5, "8") and not answer_matches_contract(p5, "8 نفر")
    assert canonicalize(p5, "8.0") == "8"
    p7 = _problem("public_007")
    assert canonicalize(p7, "monday") == "Monday"
    assert not answer_matches_contract(p7, "Friday")
    p8 = _problem("public_008")
    assert canonicalize(p8, "Ali=Lie,Bita=Truth") == "Ali=Lie, Bita=Truth"
    assert answer_matches_contract(p8, "Ali=Lie, Bita=Truth")
    p10 = _problem("public_010")
    assert answer_matches_contract(p10, "413") and not answer_matches_contract(p10, "41")
    p3 = _problem("public_003")
    assert format_hint(p3) == ""  # free-form; no false constraint


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("unit tests passed")
