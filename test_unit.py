"""Local checks that do not call the model."""

from agent import answer_matches_contract, normalize_digits, parse_solver
from sandbox import SandboxError, run_python


def test_digits() -> None:
    assert normalize_digits("۷۲ و ٢٤") == "72 و 24"


def test_parse_json() -> None:
    text = '{"answer": "8", "python": "print(8)\\n"}'
    answer, code = parse_solver(text)
    assert answer == "8"
    assert "print(8)" in code


def test_sandbox_ok() -> None:
    assert run_python("print(40 - (24 + 18 - 10))\n") == "8"


def test_sandbox_blocks_subprocess() -> None:
    try:
        run_python("import subprocess\nprint(1)\n")
    except SandboxError:
        return
    raise AssertionError("subprocess was allowed")


def test_contract_integer() -> None:
    problem = "چند دانشجو هیچ‌کدام را نمی‌خوانند؟ فقط عدد صحیح را برگردانید."
    assert answer_matches_contract(problem, "8")
    assert not answer_matches_contract(problem, "8 نفر")


def test_sandbox_rejects_multiline() -> None:
    try:
        run_python("print(1)\nprint(2)\n")
    except SandboxError:
        return
    raise AssertionError("multiline stdout was accepted")


if __name__ == "__main__":
    test_digits()
    test_parse_json()
    test_sandbox_ok()
    test_sandbox_blocks_subprocess()
    test_contract_integer()
    test_sandbox_rejects_multiline()
    print("unit tests passed")
