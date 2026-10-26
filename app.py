"""English interface for operations on finite families of conditional events."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logic import (  # noqa: E402
    ConditionalEvent,
    FamilyOperationResult,
    compute_family_operation,
    iterated_interval,
)


st.set_page_config(page_title="Conditional Events Operations", page_icon="∧", layout="wide")


def _format_coefficient(value: float) -> str:
    return "1" if abs(value - 1.0) <= 1e-12 else f"{value:.6g}"


def _system_latex(result: FamilyOperationResult, marker: str) -> str:
    equations = []
    for coordinate, event in enumerate(result.events):
        terms = []
        for number, constituent in enumerate(result.constituents, start=1):
            value = constituent.q[coordinate]
            if abs(value) <= 1e-12:
                continue
            terms.append(
                rf"{_format_coefficient(value)}\lambda{marker}_{{{number}}}"
            )
        equations.append(
            (" + ".join(terms) or "0") + rf" &= {event.probability:.6g}"
        )
    count = len(result.constituents)
    equations.append(rf"\lambda{marker}_1+\cdots+\lambda{marker}_{{{count}}} &= 1")
    equations.append(rf"\lambda{marker}_k &\geq 0,\quad k=1,\ldots,{count}")
    rows = r" \\[4pt] ".join(equations)
    return rf"\left\{{\begin{{aligned}}{rows}\end{{aligned}}\right."


def _objective_latex(result: FamilyOperationResult, marker: str, operator: str) -> str:
    terms = []
    for number, value in enumerate(result.operation_values, start=1):
        if value is None or abs(value) <= 1e-12:
            continue
        terms.append(rf"{_format_coefficient(value)}\lambda{marker}_{{{number}}}")
    objective = " + ".join(terms) or "0"
    return rf"{result.symbol}{marker}={operator}\left({objective}\right)"


def _show_constituents(result: FamilyOperationResult) -> None:
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
                "operation value": "—" if value is None else value,
                "witness": str(constituent.witnesses[0]),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_operation(result: FamilyOperationResult) -> None:
    st.success(
        f"Coherent interval for {result.symbol}: "
        f"[{result.lower:.6f}, {result.upper:.6f}]"
    )
    st.subheader("Conditional constituents")
    st.caption(
        "C₀, where every conditioning event is false, is void and is excluded "
        "from the displayed system. T, F, and V mean true, false, and void."
    )
    _show_constituents(result)

    st.subheader("Optimization defining the bounds")
    if result.exact_system:
        st.latex(_objective_latex(result, "'", r"\min"))
        st.latex(_objective_latex(result, "''", r"\max"))
        st.caption(
            "The same feasibility constraints are used for both bounds; only "
            "the objective and the lambda markers change."
        )
        st.markdown("Constraints for the lower bound:")
        st.latex(_system_latex(result, "'"))
        st.markdown("Constraints for the upper bound:")
        st.latex(_system_latex(result, "''"))
    elif result.operation == "conjunction":
        st.latex(r"z'=\max\{p_1+\cdots+p_n-n+1,0\},\qquad z''=\min_i p_i")
        st.info(
            "For more than two conditional events these are the generalized "
            "sharp Frechet--Hoeffding bounds under logical independence."
        )
    else:
        st.latex(r"w'=\max_i p_i,\qquad w''=\min\{p_1+\cdots+p_n,1\}")
        st.info(
            "For more than two conditional events these are the generalized "
            "sharp Frechet--Hoeffding bounds under logical independence."
        )


st.title("Operations between conditional events")
st.markdown(
    "This Streamlit application studies conjunction, disjunction, and iterated "
    "conditioning in the coherence framework."
)
st.latex(
    r"E_i|H_i\in\{1,0,p_i\}\quad\text{according as it is true, false, or void.}"
)
st.latex(
    r"\mathcal{C}_{1\cdots n}=\bigwedge_{i=1}^n(E_i|H_i),\qquad "
    r"\mathcal{D}_{1\cdots n}=\bigvee_{i=1}^n(E_i|H_i)"
)

st.subheader("1. Enter a finite family of conditional events")
number_of_events = st.number_input(
    "Number of conditional events",
    min_value=2,
    max_value=6,
    value=2,
    step=1,
)
default_events = [
    ("A", "H", 0.6),
    ("B", "K", 0.4),
    ("C", "L", 0.5),
    ("D", "M", 0.5),
    ("E", "N", 0.5),
    ("F", "O", 0.5),
]
events = []
for index in range(int(number_of_events)):
    event_default, condition_default, probability_default = default_events[index]
    left, middle, right = st.columns((2, 2, 1))
    with left:
        event_formula = st.text_input(
            f"Event E{index + 1}", value=event_default, key=f"event_{index}"
        )
    with middle:
        condition_formula = st.text_input(
            f"Condition H{index + 1}",
            value=condition_default,
            key=f"condition_{index}",
        )
    with right:
        probability = st.number_input(
            f"P(E{index + 1}|H{index + 1})",
            min_value=0.0,
            max_value=1.0,
            value=probability_default,
            step=0.05,
            format="%.4f",
            key=f"probability_{index}",
        )
    events.append(ConditionalEvent(event_formula, condition_formula, float(probability)))

st.subheader("2. Logical relations")
relations_text = st.text_area(
    "One relation per line",
    value="",
    help="Use and, or, not, parentheses, and optionally = True or = False.",
)
st.caption(
    "Example: `not (A and B)` means that A and B are incompatible. "
    "For n>2, the generalized bounds require logical independence, so leave "
    "this field empty when using three or more conditional events."
)
relations = tuple(line.strip() for line in relations_text.splitlines() if line.strip())

st.subheader("3. Select an operation")
operation_label = st.selectbox(
    "Operation",
    (
        "Conjunction of all conditional events",
        "Disjunction of all conditional events",
        "Iterated conditional (E2|H2)|(E1|H1)",
    ),
)

if st.button("Compute coherent bounds", type="primary"):
    try:
        if operation_label.startswith("Iterated"):
            if len(events) != 2:
                raise ValueError(
                    "Iterated conditioning is currently defined for the first two "
                    "conditional events only; set the number to 2."
                )
            conjunction = compute_family_operation(events, "conjunction", relations)
            lower, upper = iterated_interval(conjunction, events[0].probability)
            st.success(f"Coherent interval for μ: [{lower:.6f}, {upper:.6f}]")
            st.latex(r"\mu=\mathbb{P}[(E_2|H_2)|(E_1|H_1)],\qquad z=\mu\,p_1")
            st.markdown(
                "When p₁>0, the iterated bounds are z′/p₁ and z″/p₁. "
                "When p₁=0, coherence allows μ in [0,1]."
            )
            _render_operation(conjunction)
        else:
            operation = "conjunction" if operation_label.startswith("Conjunction") else "disjunction"
            _render_operation(compute_family_operation(events, operation, relations))
    except Exception as error:
        st.error(f"Input error: {error}")
