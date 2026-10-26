"""Coherence calculations for operations on finite families of conditional events."""

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
    statuses: tuple[str, ...]
    witnesses: tuple[dict[str, bool], ...]
    q: tuple[float, ...]


class FamilyOperationResult(NamedTuple):
    events: tuple[ConditionalEvent, ...]
    operation: str
    symbol: str
    lower: float
    upper: float
    constituents: tuple[ConditionalConstituent, ...]
    operation_values: tuple[float | None, ...]
    lower_lambda: np.ndarray | None
    upper_lambda: np.ndarray | None
    exact_system: bool


def _parse(formula: str) -> ast.AST:
    formula = formula.strip() or "True"
    try:
        tree = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid Boolean formula {formula!r}.") from exc
    allowed = (
        ast.Name, ast.Constant, ast.BoolOp, ast.And, ast.Or,
        ast.UnaryOp, ast.Not, ast.Load,
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
    events: Sequence[ConditionalEvent],
    relations: Sequence[str] = (),
) -> tuple[ConditionalConstituent, ...]:
    if not events:
        raise ValueError("Enter at least one conditional event.")
    formulas = [item for event in events for item in (event.event, event.condition)]
    trees = [_parse(formula) for formula in formulas]
    relation_trees = [_parse_relation(relation) for relation in relations if relation.strip()]
    primitive_names = set().union(*(_names(formula) for formula in formulas))
    primitive_names.update(
        node.id
        for tree in relation_trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    )

    groups: dict[tuple[str, ...], list[dict[str, bool]]] = {}
    for values in product((False, True), repeat=len(primitive_names)):
        assignment = dict(zip(sorted(primitive_names), values))
        if any(not _evaluate(tree, assignment) for tree in relation_trees):
            continue
        evaluated = [_evaluate(tree, assignment) for tree in trees]
        statuses = tuple(
            _conditional_status(evaluated[2 * index], evaluated[2 * index + 1])
            for index in range(len(events))
        )
        if all(status == "V" for status in statuses):
            continue  # C_0: all conditioning events are false.
        groups.setdefault(statuses, []).append(assignment)

    def status_key(statuses: tuple[str, ...]) -> tuple[int, ...]:
        return tuple({"T": 0, "F": 1, "V": 2}[status] for status in statuses)

    constituents = []
    for statuses, witnesses in groups.items():
        constituents.append(
            ConditionalConstituent(
                statuses=statuses,
                witnesses=tuple(witnesses),
                q=tuple(
                    _conditional_value(status, event.probability)
                    for status, event in zip(statuses, events)
                ),
            )
        )
    return tuple(sorted(constituents, key=lambda item: status_key(item.statuses)))


def _pair_operation_value(
    statuses: tuple[str, str], operation: str, x: float, y: float
) -> float:
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


def _solve_pair(
    events: tuple[ConditionalEvent, ConditionalEvent],
    constituents: tuple[ConditionalConstituent, ...],
    operation: str,
) -> FamilyOperationResult:
    if not constituents:
        raise ValueError("Every constituent is C_0: all conditioning events are impossible.")
    q = np.asarray([constituent.q for constituent in constituents], dtype=float)
    probabilities = np.asarray([event.probability for event in events], dtype=float)
    A_eq = np.vstack([q.T, np.ones(len(constituents))])
    b_eq = np.asarray([*probabilities, 1.0], dtype=float)
    values = np.asarray(
        [
            _pair_operation_value(
                constituent.statuses,
                operation,
                events[0].probability,
                events[1].probability,
            )
            for constituent in constituents
        ],
        dtype=float,
    )
    lower = linprog(values, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, None)] * len(constituents), method="highs")
    upper = linprog(-values, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0.0, None)] * len(constituents), method="highs")
    if not lower.success or not upper.success or lower.x is None or upper.x is None:
        raise ValueError(
            "The conditional assessment is incoherent or the operation has no "
            "feasible coherent extension."
        )
    return FamilyOperationResult(
        events=events,
        operation=operation,
        symbol="z" if operation == "conjunction" else "w",
        lower=float(lower.fun),
        upper=float(-upper.fun),
        constituents=constituents,
        operation_values=tuple(float(value) for value in values),
        lower_lambda=lower.x,
        upper_lambda=upper.x,
        exact_system=True,
    )


def compute_family_operation(
    events: Sequence[ConditionalEvent],
    operation: str,
    relations: Sequence[str] = (),
) -> FamilyOperationResult:
    """Compute bounds for the conjunction/disjunction of a finite family."""
    if operation not in {"conjunction", "disjunction"}:
        raise ValueError("Choose conjunction or disjunction.")
    family = tuple(events)
    if len(family) < 2:
        raise ValueError("Enter at least two conditional events for this operation.")
    constituents = build_constituents(family, relations)
    if len(family) == 2:
        return _solve_pair((family[0], family[1]), constituents, operation)
    if relations:
        raise ValueError(
            "For more than two events, the generalized Frechet--Hoeffding "
            "bounds require logical independence; remove the logical relations."
        )
    probabilities = [event.probability for event in family]
    if operation == "conjunction":
        lower = max(sum(probabilities) - len(family) + 1, 0.0)
        upper = min(probabilities)
        symbol = "z"
    else:
        lower = max(probabilities)
        upper = min(sum(probabilities), 1.0)
        symbol = "w"
    values: list[float | None] = []
    for constituent in constituents:
        statuses = constituent.statuses
        if operation == "conjunction":
            values.append(0.0 if "F" in statuses else 1.0 if all(status == "T" for status in statuses) else None)
        else:
            values.append(1.0 if "T" in statuses else 0.0 if all(status == "F" for status in statuses) else None)
    return FamilyOperationResult(
        events=family,
        operation=operation,
        symbol=symbol,
        lower=float(lower),
        upper=float(upper),
        constituents=constituents,
        operation_values=tuple(values),
        lower_lambda=None,
        upper_lambda=None,
        exact_system=False,
    )


def iterated_interval(conjunction_result: FamilyOperationResult, x: float) -> tuple[float, float]:
    """Return bounds for (B|K)|(A|H), using z = x * mu."""
    if x <= 1e-12:
        return 0.0, 1.0
    return conjunction_result.lower / x, conjunction_result.upper / x
