"""Program-of-Thought agent: dual solver + adjudicator + direct-reasoning fallback.

The model does not hand back the graded answer directly. It writes a Python
program, the program is executed locally, and a second independent program must
agree. Disagreement triggers one adjudication program that sees both attempts.
If no program can be produced, a direct chain-of-thought call answers instead,
so the output is never empty while budget remains.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List, Optional, Tuple

from llm import DEFAULT_MODEL, chat, parse_json_object, resolve_model
from sandbox import SandboxError, run_python

MAX_LLM_CALLS = int(os.environ.get("AGENT_MAX_CALLS", "5"))
TIME_BUDGET = float(os.environ.get("AGENT_TIME_BUDGET", "600"))  # seconds per problem

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
- Match the problem's required format exactly: spelling, capitalization, order, commas, and spaces. If the problem lists allowed phrases, print one of them verbatim. If it gives a template such as Name=<integer>, replace only the placeholders.
- The problem may be in Persian or English. Integers must use ASCII digits.

Return only a JSON object with two string fields:
{"answer": "<the exact line the program prints>", "python": "<complete program>"}
The program's printed line must equal "answer".
"""

REPAIR_SYSTEM = """You fix a Python 3 solver that failed or disagreed with another solver.
Return only a JSON object:
{"answer": "<exact final line>", "python": "<complete corrected program>"}
The program must print exactly that one line, use only the standard library, and encode every constraint in the problem. Do not print explanations.
"""

DIRECT_SYSTEM = """You are a careful reasoning assistant. Solve the problem step by step,
check the result against every constraint, then give the final answer.
Return only a JSON object: {"reasoning": "<short step-by-step work>", "answer": "<final answer only>"}
The answer must follow the problem's required format exactly, with no labels, units or explanation.
"""


def normalize_digits(text: str) -> str:
    persian = "۰۱۲۳۴۵۶۷۸۹"
    arabic = "٠١٢٣٤٥٦٧٨٩"
    table = {ord(src): str(i) for i, src in enumerate(persian)}
    table.update({ord(src): str(i) for i, src in enumerate(arabic)})
    table[ord("٫")] = "."  # Arabic decimal separator
    table[ord("٬")] = ","  # Arabic thousands separator
    return text.translate(table)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _extract_code_block(text: str) -> str:
    fenced = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return max(fenced, key=len).strip()
    return ""


def parse_solver(text: str) -> Tuple[str, str]:
    data = parse_json_object(text)
    answer = ""
    code = ""
    if data:
        raw_answer = data.get("answer", "")
        raw_code = data.get("python") or data.get("code") or ""
        if isinstance(raw_answer, bool):
            answer = str(raw_answer)
        elif isinstance(raw_answer, (int, float)):
            answer = str(int(raw_answer)) if float(raw_answer).is_integer() else str(raw_answer)
        elif isinstance(raw_answer, str):
            answer = raw_answer.strip()
        if isinstance(raw_code, str):
            code = raw_code.strip()
    if code.startswith("```"):
        code = _extract_code_block(code + "\n```") or code
    if not code:
        code = _extract_code_block(text)
    if not answer:
        match = re.search(r"(?im)^answer\s*[:=]\s*(.+)$", text)
        if match:
            answer = match.group(1).strip().strip("`")
    if answer.startswith("```"):
        answer = ""
    return answer, code


# ---------------------------------------------------------------- output contract

_TEMPLATE_SLOT = re.compile(r"<\s*(integer|int|number|digit|letter|word|name|day)\s*>?", re.IGNORECASE)
_SLOT_REGEX = {
    "integer": r"-?\d+",
    "int": r"-?\d+",
    "number": r"-?\d+(?:\.\d+)?",
    "digit": r"\d",
    "letter": r"[A-Za-z]",
    "word": r"[A-Za-z]+",
    "name": r"[A-Za-z]+",
    "day": r"[A-Za-z]+",
}


def _find_template(problem: str) -> Optional[str]:
    """Find a line like `Nima=<integer>, Sara=<integer>, Reza=<integer>`."""
    for line in problem.splitlines() or [problem]:
        for m in re.finditer(r"<?[A-Za-z][\w ]*\s*[=:]\s*<\s*\w+\s*>?(?:\s*[,;]\s*<?[A-Za-z][\w ]*\s*[=:]\s*<\s*\w+\s*>?)+", line):
            return m.group(0).strip()
    return None


def _template_regex(template: str) -> str:
    # Repair stray `<` before a key (seen in PDF copy/paste) and a missing closing `>`.
    template = re.sub(r"^<(?=[A-Za-z])", "", template.strip())
    parts: List[str] = []
    pos = 0
    for m in _TEMPLATE_SLOT.finditer(template):
        literal = template[pos : m.start()]
        parts.append(re.escape(literal).replace(r"\ ", r"\s*").replace(",", r",\s*"))
        parts.append(_SLOT_REGEX[m.group(1).lower()])
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    return "".join(parts)


def _choices(problem: str) -> List[str]:
    choice = re.search(
        r"(یکی از این|یکی از گزینه|یکی از عبارت|exactly one of|one of the following)",
        problem,
        flags=re.IGNORECASE,
    )
    if not choice:
        return []
    tail = problem[choice.end() :]
    lines = re.findall(r"(?m)^\s*[-•*]?\s*([A-Za-z][A-Za-z0-9 =,:_\-]{0,80})\s*$", tail)
    return [line.strip() for line in lines if line.strip()]


_NUM_WORDS = {"دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6,
              "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _code_length(problem: str) -> int:
    """Digits required when the answer is an N-digit code, e.g. 'فقط کد سه‌رقمی را برگردانید'."""
    m = re.search(r"فقط\s+(?:کد|عدد)\s+(\d|دو|سه|چهار|پنج|شش)[\s\u200c-]*رقمی", problem)
    if not m:
        m = re.search(r"only\s+the\s+(\d|two|three|four|five|six)[\s-]*digit", problem, flags=re.IGNORECASE)
    if not m:
        return 0
    word = m.group(1).lower()
    return int(word) if word.isdigit() else _NUM_WORDS[word]


def format_hint(problem: str) -> str:
    """Describe the answer line when the problem states a strict format."""
    hints: List[str] = []
    if re.search(r"فقط عدد صحیح|عدد صحیح را برگردان|only (?:an |the )?integer|return only the integer",
                 problem, flags=re.IGNORECASE):
        hints.append("The single printed line must match ^-?[0-9]+$")
    if re.search(r"فقط حرف|only the letter|single letter", problem, flags=re.IGNORECASE):
        hints.append("The single printed line must be exactly one letter.")
    n = _code_length(problem)
    if n:
        hints.append(f"The single printed line must match ^[0-9]{{{n}}}$")
    template = _find_template(problem)
    if template:
        hints.append(f"The printed line must follow this template exactly, replacing the placeholders: {template}")
    listed = _choices(problem)
    if listed:
        hints.append("The printed line must be exactly one of: " + " | ".join(listed))
    if not hints:
        return ""
    return "Output contract:\n" + "\n".join(f"- {hint}" for hint in hints)


def canonicalize(problem: str, answer: str) -> str:
    """Strip harmless decoration and snap near-misses onto the required format."""
    a = normalize_digits(str(answer)).strip()
    a = re.sub(r"^(?:final answer|answer|پاسخ(?: نهایی)?)\s*[:：=]\s*", "", a, flags=re.IGNORECASE)
    a = a.strip().strip("`*\"'").strip()
    a = a.rstrip(".。").strip()
    hint = format_hint(problem)
    if "^-?[0-9]+$" in hint:
        m = re.fullmatch(r"(-?\d+)(?:\.0+)?", a.replace(",", ""))
        if m:
            a = m.group(1)
    for choice in _choices(problem):
        if re.sub(r"\s+", "", a).lower() == re.sub(r"\s+", "", choice).lower():
            return choice
    return a


def answer_matches_contract(problem: str, answer: str) -> bool:
    if not answer:
        return False
    if "^-?[0-9]+$" in format_hint(problem) and not re.fullmatch(r"-?\d+", answer):
        return False
    hint = format_hint(problem)
    if "exactly one letter" in hint and not re.fullmatch(r"[A-Za-z]", answer):
        return False
    m = re.search(r"\^\[0-9\]\{(\d)\}\$", hint)
    if m and not re.fullmatch(r"\d{%s}" % m.group(1), answer):
        return False
    template = _find_template(problem)
    if template and not re.fullmatch(_template_regex(template), answer):
        return False
    choices = _choices(problem)
    if choices and answer not in choices:
        return False
    return True


def _messages(system: str, user: str) -> List[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class BudgetExhausted(Exception):
    pass


class Solver:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls = 0
        self.started = time.monotonic()

    def _left(self) -> float:
        return TIME_BUDGET - (time.monotonic() - self.started)

    def _can_call(self, reserve: int = 0) -> bool:
        return self.calls + reserve < MAX_LLM_CALLS and self._left() > 20

    def _ask(self, system: str, user: str, *, temperature: float, seed: int,
             num_predict: int = 3000) -> str:
        if not self._can_call():
            raise BudgetExhausted("LLM call or time budget exhausted")
        self.calls += 1
        _log(f"[agent] LLM call {self.calls}/{MAX_LLM_CALLS} temperature={temperature} seed={seed}")
        try:
            return chat(
                _messages(system, user),
                model=self.model,
                temperature=temperature,
                seed=seed,
                num_predict=num_predict,
                timeout=max(30.0, self._left()),
            )
        except Exception as exc:  # timeout, HTTP error, connection refused
            _log(f"[agent] LLM call failed: {exc}")
            raise BudgetExhausted(str(exc)) from exc

    def _execute(self, problem: str, code: str) -> str:
        printed = canonicalize(problem, run_python(code))
        if not answer_matches_contract(problem, printed):
            raise SandboxError(f"printed line does not match the required format: {printed!r}")
        return printed

    def _program_once(self, problem: str, user: str, *, system: str,
                      temperature: float, seed: int) -> Optional[str]:
        try:
            text = self._ask(system, user, temperature=temperature, seed=seed)
        except BudgetExhausted:
            return None
        claimed, code = parse_solver(text)
        if not code:
            _log("[agent] model returned no program")
            return None
        try:
            printed = self._execute(problem, code)
        except SandboxError as exc:
            _log(f"[agent] program rejected: {exc}")
            # Keep one call in reserve for the direct-answer fallback.
            if not self._can_call(reserve=1):
                return None
            repair_user = (
                f"Problem:\n{problem}\n\n"
                f"{format_hint(problem)}\n\n"
                f"Failed program:\n{code}\n\n"
                f"Execution error:\n{exc}\n\n"
                "Write a corrected program."
            )
            try:
                text = self._ask(REPAIR_SYSTEM, repair_user, temperature=0, seed=seed + 17)
            except BudgetExhausted:
                return None
            _, repaired = parse_solver(text)
            if not repaired:
                return None
            try:
                printed = self._execute(problem, repaired)
            except SandboxError as again:
                _log(f"[agent] repair rejected: {again}")
                return None
        if claimed and canonicalize(problem, claimed) != printed:
            _log(f"[agent] using executed line {printed!r} instead of claimed {claimed!r}")
        return printed

    def _direct(self, problem: str, contract_block: str) -> Optional[str]:
        """Fallback: plain step-by-step reasoning when no program worked."""
        try:
            text = self._ask(
                DIRECT_SYSTEM,
                f"{problem}{contract_block}",
                temperature=0,
                seed=7,
                num_predict=4000,
            )
        except BudgetExhausted:
            return None
        data = parse_json_object(text) or {}
        raw = data.get("answer", "")
        if not raw:
            m = re.search(r"(?im)^\s*(?:final answer|answer)\s*[:=]\s*(.+)$", text)
            raw = m.group(1) if m else ""
        answer = canonicalize(problem, str(raw))
        if answer and not answer_matches_contract(problem, answer):
            _log(f"[agent] direct answer {answer!r} violates the format; keeping it as last resort")
        return answer or None

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
        first = self._program_once(problem, first_user, system=SYSTEM, temperature=0, seed=1)

        second_user = (
            "Solve this problem again with a Python program, independently. "
            "First list every constraint as a comment, then encode each one. "
            "Prefer exact arithmetic for quantities and graph search for paths.\n\n"
            f"{problem}{contract_block}"
        )
        second = None
        if self._can_call(reserve=0 if first else 1):
            second = self._program_once(problem, second_user, system=SYSTEM, temperature=0.3, seed=2)

        if first and second and first == second:
            _log("[agent] solvers agree")
            return first
        if first and second:
            _log(f"[agent] disagreement: {first!r} vs {second!r}")
            adjudicate = (
                "Two programs printed different answers for the same problem. "
                "Re-read the problem, find which constraints were modeled incorrectly, "
                "and write one corrected program.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Answer A:\n{first}\n\n"
                f"Answer B:\n{second}\n"
            )
            third = self._program_once(problem, adjudicate + contract_block,
                                       system=REPAIR_SYSTEM, temperature=0, seed=3)
            if third:
                return third
            return first
        if first or second:
            return first or second
        _log("[agent] no program succeeded; falling back to direct reasoning")
        return self._direct(problem, contract_block) or ""


def solve_problem(problem: str, model: Optional[str] = None) -> str:
    solver = Solver(resolve_model(model or DEFAULT_MODEL))
    try:
        answer = solver.solve(problem)
    except Exception as exc:
        _log(f"[agent] failed: {exc}")
        return ""
    return answer.strip()
