# Conditional Events — Operations

An English Streamlit application for studying logical operations between two
conditional events in the coherence framework.
Try the application online on [Streamlit](https://vsafswmanv9gjmxasrskbc.streamlit.app).
The first version covers:

- conjunction \\( (A|H)\\wedge(B|K) \\);
- disjunction \\( (A|H)\\vee(B|K) \\);
- iterated conditioning \\( (B|K)|(A|H) \\).

For a conditional event \\(A|H\\), the numerical values are \\(1\\), \\(0\\),
and \\(P(A|H)\\), according as \\(A|H\\) is true, false, or void. The app
enumerates the conditional constituents, associates a point with each one,
and uses linear programming to compute coherent lower and upper prevision
bounds for the selected operation.

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Boolean formulas use `and`, `or`, `not`, and parentheses. Logical relations
are entered one per line; for example, `not (A and B)` means that `A` and `B`
are incompatible.

The conjunction is defined as

\\[
(A|H)\\wedge(B|K)=\\min\\{A|H,B|K\\}\\mid(H\\vee K),
\\]

and the disjunction is obtained by De Morgan's law. For the iterated
conditional, the app uses

\\[
(B|K)|(A|H)=(B|K)\\wedge(A|H)+\\mu\\,\\overline{A}|H,
\\]

where \\(\\mu\\) is its prevision and, when \\(x=P(A|H)>0\\),
\\(z=\\mu x\\) for the conjunction prevision \\(z\\).
