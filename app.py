"""English interface for operations between conditional events."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic import (  # noqa: E402
    ConditionalEvent,
    OperationResult,
    build_constituents,
    compute_operation,
    iterated_interval,
)


st.set_page_config(page_title="Conditional Events Operations", page_icon="∧", layout="wide")


def _formula_latex(formula: str) -> str:
    """Make the interface formulas readable without trying to typeset logic fully."""
    result = formula.replace("not ", r"\overline{")
    if result != formula:
        result += "}"
    return result.replace(" and ", r"\land ").replace(" or ", r"\lor ")


def _system_latex(result: OperationResult, first: ConditionalEvent, second: ConditionalEvent, marker: str) -> str:
    equations = []
    for coordinate, probability in enumerate(
        (first.probability, second.probability)
    ):
        # The equation is most clearly written using the q-coordinate itself.
        terms = []
        for number, constituent in enumerate(result.constituents, start=1):
            value = constituent.q[coordinate]
            if abs(value) <= 1e-12:
                continue
            coefficient = "1" if abs(value - 1.0) <= 1e-12 else f"{value:.6g}"
            terms.append(rf"{coefficient}\lambda{marker}_{{{number}}}")
        equations.append((" + ".join(terms) or "0") + rf" &= {probability:.6g}")
    count = len(result.constituents)
    equations.append(rf"\lambda{marker}_1+\cdots+\lambda{marker}_{{{count}}} &= 1")
    equations.append(rf"\lambda{marker}_k &\geq 0,\quad k=1,\ldots,{count}")
    rows = r" \\[4pt] ".join(equations)
    return rf"\left\{{\begin{{aligned}}{rows}\end{{aligned}}\right."


def _objective_latex(result: OperationResult, marker: str, operator: str) -> str:
    terms = []
    for number, value in enumerate(result.operation_values, start=1):
        if abs(value) <= 1e-12:
            continue
        coefficient = "1" if abs(value - 1.0) <= 1e-12 else f"{value:.6g}"
        terms.append(rf"{coefficient}\lambda{marker}_{{{number}}}")
    objective = " + ".join(terms) or "0"
    return rf"{result.symbol}{marker}={operator}\left({objective}\right)"


def _show_constituents(result: OperationResult) -> None:
    rows = []
    for number, (constituent, value) in enumerate(
        zip(result.constituents, result.operation_values), start=1
    ):
        rows.append(
            {
                "constituent": f"C{number}",
                "conditional statuses": " / ".join(
                    {"T": "true", "F": "false", "V": "void"}[status]
                    for status in constituent.statuses
                ),
                "q": constituent.q,
                "operation value": value,
                "witness": str(constituent.witnesses[0]),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_operation(result: OperationResult, first: ConditionalEvent, second: ConditionalEvent) -> None:
    st.success(
        f"Coherent interval for {result.symbol}: "
        f"[{result.lower:.6f}, {result.upper:.6f}]"
    )
    st.subheader("Conditional constituents")
    st.caption(
        "C₀, where both conditioning events are false, is void and is excluded "
        "from the displayed system. T, F, and V mean true, false, and void."
    )
    _show_constituents(result)

    st.subheader("Optimization defining the bounds")
    st.latex(_objective_latex(result, "'", r"\min"))
    st.latex(_objective_latex(result, "''", r"\max"))
    st.caption(
        "The same feasibility constraints are used for the lower and upper bound; "
        "only the objective and the lambda markers change."
    )
    st.markdown("Constraints for the lower bound:")
    st.latex(_system_latex(result, first, second, "'"))
    st.markdown("Constraints for the upper bound:")
    st.latex(_system_latex(result, first, second, "''"))


st.title("Operations between conditional events")
st.markdown(
    "This Streamlit application studies conjunction, disjunction, and iterated "
    "conditioning in the coherence framework."
)
st.latex(r"A|H\in\{1,0,P(A|H)\}\quad\text{according as it is true, false, or void.}")
st.markdown(
    "The conjunction is defined by conditioning the minimum on the event "
    "that at least one of the two conditioning events occurs."
)
st.latex(r"(A|H)\wedge(B|K)=\min\{A|H,B|K\}\mid(H\vee K)")
st.latex(r"(A|H)\vee(B|K)=\bigl[(\overline{A}|H)\wedge(\overline{B}|K)\bigr]^c")
st.latex(r"(B|K)|(A|H)=(B|K)\wedge(A|H)+\mu\,\overline{A}|H")

st.subheader("1. Enter two conditional events")
left, right = st.columns(2)
with left:
    st.markdown("**First conditional event**")
    first_event = st.text_input("Event A", value="A")
    first_condition = st.text_input("Condition H", value="H")
    first_probability = st.number_input("P(A|H)", 0.0, 1.0, 0.6, 0.05, format="%.4f")
with right:
    st.markdown("**Second conditional event**")
    second_event = st.text_input("Event B", value="B")
    second_condition = st.text_input("Condition K", value="K")
    second_probability = st.number_input("P(B|K)", 0.0, 1.0, 0.4, 0.05, format="%.4f")

st.subheader("2. Logical relations")
relations_text = st.text_area(
    "One relation per line",
    value="",
    help="Use and, or, not, parentheses, and optionally = True or = False.",
)
st.caption("Example: `not (A and B)` means that A and B are incompatible.")

first = ConditionalEvent(first_event, first_condition, float(first_probability))
second = ConditionalEvent(second_event, second_condition, float(second_probability))
relations = tuple(line.strip() for line in relations_text.splitlines() if line.strip())

st.subheader("3. Select an operation")
operation_label = st.selectbox(
    "Operation",
    (
        "Conjunction (A|H) ∧ (B|K)",
        "Disjunction (A|H) ∨ (B|K)",
        "Iterated conditional (B|K)|(A|H)",
    ),
)

if st.button("Compute coherent bounds", type="primary"):
    try:
        if operation_label.startswith("Conjunction"):
            result = compute_operation(first, second, "conjunction", relations)
            _render_operation(result, first, second)
        elif operation_label.startswith("Disjunction"):
            result = compute_operation(first, second, "disjunction", relations)
            _render_operation(result, first, second)
        else:
            conjunction = compute_operation(first, second, "conjunction", relations)
            lower, upper = iterated_interval(conjunction, first.probability)
            st.success(f"Coherent interval for μ: [{lower:.6f}, {upper:.6f}]")
            st.latex(r"\mu=\mathbb{P}[(B|K)|(A|H)],\qquad z=\mu\,P(A|H)")
            st.markdown(
                "The iterated conditional is obtained from the conjunction "
                "prevision. When P(A|H)>0, its bounds are z′/P(A|H) and "
                "z″/P(A|H). When P(A|H)=0, coherence allows μ in [0,1]."
            )
            _render_operation(conjunction, first, second)
    except Exception as error:
        st.error(f"Input error: {error}")
