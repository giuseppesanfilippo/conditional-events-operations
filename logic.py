"""Coherence calculations for elementary operations on conditional events."""

from __future__ import annotations

import ast
import re
from itertools import product
from typing import NamedTuple, Sequence

import numpy as np
from scipy.optimize import linprog


class ConditionalEvent(NamedTuple):
    event: str
    condition: str
    probability: float


class ConditionalConstituent(NamedTuple):
    statuses: tuple[str, str]
    witnesses: tuple[dict[str, bool], ...]
    q: tuple[float, float]


class OperationResult(NamedTuple):
    operation: str
    symbol: str
    lower: float
    upper: float
    constituents: tuple[ConditionalConstituent, ...]
    operation_values: tuple[float, ...]
    lower_lambda: np.ndarray
    upper_lambda: np.ndarray


STATUS_ORDER = {
    ("T", "T"): 0,
    ("T", "F"): 1,
    ("T", "V"): 2,
    ("F", "T"): 3,
    ("F", "F"): 4,
    ("F", "V"): 5,
    ("V", "T"): 6,
    ("V", "F"): 7,
}


def _parse(formula: str) -> ast.AST:
    formula = formula.strip() or "True"
    try:
        tree = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid Boolean formula {formula!r}.") from exc
    allowed = (
        ast.Name,
        ast.Constant,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Load,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError(
            f"Unsupported syntax in {formula!r}. Use and, or, not, and parentheses."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            raise ValueError(f"Only Boolean constants are allowed in {formula!r}.")
    return tree


def _names(formula: str) -> set[str]:
    return {node.id for node in ast.walk(_parse(formula)) if isinstance(node, ast.Name)}


def _evaluate(tree: ast.AST, assignment: dict[str, bool]) -> bool:
    if isinstance(tree, ast.Name):
        return assignment[tree.id]
    if isinstance(tree, ast.Constant):
        return bool(tree.value)
    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not):
        return not _evaluate(tree.operand, assignment)
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        return all(_evaluate(value, assignment) for value in tree.values)
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.Or):
        return any(_evaluate(value, assignment) for value in tree.values)
    raise ValueError("Unsupported Boolean expression.")


def _parse_relation(relation: str) -> ast.AST:
    relation = relation.strip()
    implication = re.fullmatch(r"(.+?)\s+implies\s+(.+)", relation, flags=re.I)
    if implication:
        left = _parse(implication.group(1))
        right = _parse(implication.group(2))
        return ast.BoolOp(
            op=ast.Or(),
            values=[ast.UnaryOp(op=ast.Not(), operand=left), right],
        )
    match = re.fullmatch(r"(.+?)\s*=\s*(True|False)", relation, flags=re.I)
    if match:
        tree = _parse(match.group(1))
        return tree if match.group(2).lower() == "true" else ast.UnaryOp(
            op=ast.Not(), operand=tree
        )
    return _parse(relation)


def _conditional_status(event: bool, condition: bool) -> str:
    if not condition:
        return "V"
    return "T" if event else "F"


def _conditional_value(status: str, probability: float) -> float:
    return {"T": 1.0, "F": 0.0, "V": probability}[status]


def build_constituents(
    first: ConditionalEvent,
    second: ConditionalEvent,
    relations: Sequence[str] = (),
) -> tuple[ConditionalConstituent, ...]:
    formulas = [first.event, first.condition, second.event, second.condition]
    trees = [_parse(formula) for formula in formulas]
    relation_trees = [_parse_relation(relation) for relation in relations if relation.strip()]
    primitive_names = sorted(set().union(*(_names(formula) for formula in formulas)))
    primitive_names = sorted(
        set(primitive_names).union(
            *(
                {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                for tree in relation_trees
            )
        )
    )
    groups: dict[tuple[str, str], list[dict[str, bool]]] = {}
    for values in product((False, True), repeat=len(primitive_names)):
        assignment = dict(zip(primitive_names, values))
        if any(not _evaluate(tree, assignment) for tree in relation_trees):
            continue
        first_event, first_condition, second_event, second_condition = (
            _evaluate(tree, assignment) for tree in trees
        )
        statuses = (
            _conditional_status(first_event, first_condition),
            _conditional_status(second_event, second_condition),
        )
        if statuses == ("V", "V"):
            continue  # C_0: all conditioning events are false.
        groups.setdefault(statuses, []).append(assignment)
    constituents = [
        ConditionalConstituent(
            statuses=statuses,
            witnesses=tuple(witnesses),
            q=(
                _conditional_value(statuses[0], first.probability),
                _conditional_value(statuses[1], second.probability),
            ),
        )
        for statuses, witnesses in groups.items()
    ]
    return tuple(
        sorted(constituents, key=lambda constituent: STATUS_ORDER[constituent.statuses])
    )


def _operation_value(
    statuses: tuple[str, str],
    operation: str,
    x: float,
    y: float,
) -> float:
    first_status, second_status = statuses
    if operation == "conjunction":
        if "F" in statuses:
            return 0.0
        if statuses == ("T", "T"):
            return 1.0
        if statuses == ("V", "T"):
            return x
        if statuses == ("T", "V"):
            return y
    if operation == "disjunction":
        if "T" in statuses:
            return 1.0
        if statuses == ("F", "F"):
            return 0.0
        if statuses == ("V", "F"):
            return x
        if statuses == ("F", "V"):
            return y
    raise ValueError(f"Unsupported operation state {statuses!r}.")


def _solve_bounds(
    first: ConditionalEvent,
    second: ConditionalEvent,
    constituents: tuple[ConditionalConstituent, ...],
    operation: str,
) -> OperationResult:
    if not constituents:
        raise ValueError("Every constituent is C_0: both conditioning events are impossible.")
    q = np.asarray([constituent.q for constituent in constituents], dtype=float)
    p = np.asarray([first.probability, second.probability], dtype=float)
    A_eq = np.vstack([q.T, np.ones(len(constituents))])
    b_eq = np.asarray([*p, 1.0], dtype=float)
    values = np.asarray(
        [
            _operation_value(
                constituent.statuses,
                operation,
                first.probability,
                second.probability,
            )
            for constituent in constituents
        ],
        dtype=float,
    )
    lower = linprog(
        values,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * len(constituents),
        method="highs",
    )
    upper = linprog(
        -values,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * len(constituents),
        method="highs",
    )
    if not lower.success or not upper.success or lower.x is None or upper.x is None:
        raise ValueError(
            "The original conditional assessment is incoherent or the operation "
            "has no feasible extension."
        )
    symbol = "z" if operation == "conjunction" else "w"
    return OperationResult(
        operation=operation,
        symbol=symbol,
        lower=float(lower.fun),
        upper=float(-upper.fun),
        constituents=constituents,
        operation_values=tuple(float(value) for value in values),
        lower_lambda=lower.x,
        upper_lambda=upper.x,
    )


def compute_operation(
    first: ConditionalEvent,
    second: ConditionalEvent,
    operation: str,
    relations: Sequence[str] = (),
) -> OperationResult:
    if operation not in {"conjunction", "disjunction"}:
        raise ValueError("The operation must be conjunction or disjunction.")
    constituents = build_constituents(first, second, relations)
    return _solve_bounds(first, second, constituents, operation)


def iterated_interval(conjunction_result: OperationResult, x: float) -> tuple[float, float]:
    """Return bounds for (B|K)|(A|H), using z = x * mu."""
    if x <= 1e-12:
        return 0.0, 1.0
    return conjunction_result.lower / x, conjunction_result.upper / x
