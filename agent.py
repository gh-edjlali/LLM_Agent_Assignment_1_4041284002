"""Planner / dual solver / adjudicator agent.

The model never returns the graded answer directly. It writes a Python program,
the program is executed locally, and a second independent program must agree.
Disagreement triggers one repair program that can see both attempts.
"""

from __future__ import annotations

import re
import sys

from llm import DEFAULT_MODEL, chat, parse_json_object
from sandbox import SandboxError, run_python

MAX_LLM_CALLS = 5

SYSTEM = """You solve reasoning problems by writing a correct Python 3 program.
The program is executed. Its printed line is the only answer that counts.

Rules for the program:
- Python 3 standard library only.
- No files, network, subprocess, eval, or exec.
- Compute the answer. Do not just print a guessed constant.
- Use exact arithmetic (fractions or integers), exhaustive search, or BFS/DFS when that fits the problem.
- Enumerate assignments, orderings, truth values, or digit codes when the search space is small.
- Obey every constraint in the problem. Do not add constraints that are not stated.
- If several full solutions exist, answer the question that was actually asked (for example a value shared by every solution). If the question asks for one specific solution and several exist, prefer the solution the constraints uniquely determine; if nothing is unique, still print one candidate that satisfies every constraint and the required format.
- Print exactly one line and nothing else. No labels such as "answer:", no markdown, no explanation, no units, unless the problem explicitly requires those characters.
- Match the problem's required format exactly: spelling, capitalization, commas, and spaces. If the problem lists allowed phrases, print one of them verbatim.
- The problem may be in Persian or English. Follow the language the problem demands for the answer. Integers must use ASCII digits.

Return only a JSON object with two string fields:
{"answer": "<the exact line the program prints>", "python": "<complete program>"}
The program's printed line must equal "answer".
"""

REPAIR_SYSTEM = """You fix a Python 3 solver that failed or disagreed with another solver.
Return only a JSON object:
{"answer": "<exact final line>", "python": "<complete corrected program>"}
The program must print exactly that one line, use only the standard library, and encode every constraint in the problem. Do not print explanations.
"""


def normalize_digits(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    table = {ord(src): str(i) for i, src in enumerate(persian)}
    table.update({ord(src): str(i) for i, src in enumerate(arabic)})
    return text.translate(table)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _extract_code_block(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return max(fenced, key=len).strip()
    return ""


def parse_solver(text: str) -> tuple[str, str]:
    data = parse_json_object(text)
    answer = ""
    code = ""
    if data:
        raw_answer = data.get("answer", "")
        raw_code = data.get("python") or data.get("code") or ""
        if isinstance(raw_answer, (int, float)):
            answer = str(raw_answer)
        elif isinstance(raw_answer, str):
            answer = raw_answer.strip()
        if isinstance(raw_code, str):
            code = raw_code.strip()
    if not code:
        code = _extract_code_block(text)
    if not answer:
        match = re.search(r"(?im)^answer\s*[:=]\s*(.+)$", text)
        if match:
            answer = match.group(1).strip().strip("`")
    if answer.startswith("```"):
        answer = ""
    return answer, code


def format_hint(problem: str) -> str:
    """Describe the answer line when the problem states a strict format."""
    hints: list[str] = []
    if re.search(
        r"فقط عدد صحیح|only (?:an |the )?integer|return only the integer",
        problem,
        flags=re.IGNORECASE,
    ):
        hints.append("The single printed line must match ^-?[0-9]+$")
    if re.search(r"فقط حرف", problem):
        hints.append("The single printed line must be exactly one letter.")
    if re.search(r"کد سه\s*رقمی|3-digit|three-digit", problem, flags=re.IGNORECASE):
        hints.append("The single printed line must match ^[0-9]{3}$")
    choice = re.search(
        r"(یکی از این|یکی از گزینه|exactly one of|one of the following)",
        problem,
        flags=re.IGNORECASE,
    )
    if choice:
        tail = problem[choice.end() :]
        options = re.findall(r"(?m)^[A-Za-z][A-Za-z0-9 =,]{0,80}$", tail)
        listed = [line.strip() for line in options if line.strip()]
        if listed:
            rendered = " | ".join(listed)
            hints.append(f"The printed line must be exactly one of: {rendered}")
    if not hints:
        return ""
    return "Output contract:\n" + "\n".join(f"- {hint}" for hint in hints)


def answer_matches_contract(problem: str, answer: str) -> bool:
    hint = format_hint(problem)
    if "^-?[0-9]+$" in hint and not re.fullmatch(r"-?\d+", answer):
        return False
    if "exactly one letter" in hint and not re.fullmatch(r"[A-Za-z]", answer):
        return False
    if "^[0-9]{3}$" in hint and not re.fullmatch(r"\d{3}", answer):
        return False
    if "exactly one of:" in hint:
        allowed = hint.split("exactly one of:", 1)[1]
        choices = [part.strip() for part in allowed.split("|")]
        choices = [part.split("\n", 1)[0].strip(" -") for part in choices]
        if answer not in choices:
            return False
    return True


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class Solver:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0

    def _ask(self, system: str, user: str, *, temperature: float, seed: int) -> str:
        if self.calls >= MAX_LLM_CALLS:
            raise RuntimeError("LLM call budget exhausted")
        self.calls += 1
        _log(f"[agent] LLM call {self.calls}/{MAX_LLM_CALLS} temperature={temperature} seed={seed}")
        return chat(
            _messages(system, user),
            model=self.model,
            temperature=temperature,
            seed=seed,
        )

    def _execute(self, problem: str, code: str) -> str:
        printed = run_python(code)
        if not answer_matches_contract(problem, printed):
            raise SandboxError(f"printed line does not match the required format: {printed!r}")
        return printed

    def _program_once(
        self,
        problem: str,
        user: str,
        *,
        system: str,
        temperature: float,
        seed: int,
    ) -> str | None:
        text = self._ask(system, user, temperature=temperature, seed=seed)
        claimed, code = parse_solver(text)
        if not code:
            _log("[agent] model returned no program")
            return None
        try:
            printed = self._execute(problem, code)
        except SandboxError as exc:
            _log(f"[agent] program rejected: {exc}")
            if self.calls >= MAX_LLM_CALLS:
                return None
            repair_user = (
                f"Problem:\n{problem}\n\n"
                f"{format_hint(problem)}\n\n"
                f"Failed program:\n{code}\n\n"
                f"Execution error:\n{exc}\n\n"
                "Write a corrected program."
            )
            text = self._ask(REPAIR_SYSTEM, repair_user, temperature=0, seed=seed + 17)
            _, repaired = parse_solver(text)
            if not repaired:
                return None
            try:
                printed = self._execute(problem, repaired)
            except SandboxError as again:
                _log(f"[agent] repair rejected: {again}")
                return None
        if claimed and claimed != printed:
            _log(f"[agent] using executed line {printed!r} instead of claimed {claimed!r}")
        return printed

    def solve(self, problem: str) -> str:
        problem = normalize_digits(problem).strip()
        if not problem:
            return ""
        contract = format_hint(problem)
        contract_block = f"\n\n{contract}" if contract else ""

        first_user = (
            "Solve this problem by writing an executable Python program. "
            "For constraint puzzles, schedules, assignments, and codes, exhaustively "
            "enumerate the legal candidates and keep only those that satisfy every constraint.\n\n"
            f"{problem}{contract_block}"
        )
        first = self._program_once(
            problem, first_user, system=SYSTEM, temperature=0, seed=1
        )

        second_user = (
            "Solve this problem again with a Python program, independently. "
            "Prefer exact arithmetic for quantities and graph search for paths. "
            "Re-read every constraint before accepting a candidate.\n\n"
            f"{problem}{contract_block}"
        )
        second: str | None = None
        if self.calls < MAX_LLM_CALLS:
            second = self._program_once(
                problem, second_user, system=SYSTEM, temperature=0.3, seed=2
            )

        if first and second and first == second:
            _log("[agent] solvers agree")
            return first
        if first and second and first != second and self.calls < MAX_LLM_CALLS:
            _log(f"[agent] disagreement: {first!r} vs {second!r}")
            adjudicate = (
                "Two programs printed different answers for the same problem. "
                "Find which constraints were modeled incorrectly and write one corrected program.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Answer A:\n{first}\n\n"
                f"Answer B:\n{second}\n"
            )
            third = self._program_once(
                problem,
                adjudicate + contract_block,
                system=REPAIR_SYSTEM,
                temperature=0,
                seed=3,
            )
            if third:
                if third == first or third == second:
                    return third
                return third
        for candidate in (first, second):
            if candidate:
                return candidate
        return ""


def solve_problem(problem: str, model: str | None = None) -> str:
    solver = Solver(model or DEFAULT_MODEL)
    try:
        answer = solver.solve(problem)
    except Exception as exc:
        _log(f"[agent] failed: {exc}")
        return ""
    return answer.strip()
