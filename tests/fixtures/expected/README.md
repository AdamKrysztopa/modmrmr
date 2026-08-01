# Expected regression fixtures

The JSON files in this directory are **frozen characterization fixtures** for
the lag-aware ModMRMR scenarios in `modmrmr.diagnostics.regression`. They make
changes to previously captured engine output visible. The regression test
compares only ModMRMR-owned fields; downstream `contract`, `lag_table`, and
`markdown_length` fields are outside its scope.

## Oracle status

There is currently no documented independent origin or review record for these
expected values. Treat them as change-detection evidence, not as proof that the
captured behavior is mathematically or product-correct. A passing test means
that the implementation reproduces the frozen behavior. It does not, by
itself, validate that behavior against a specification, analytic result, or
independent reference implementation.

All JSON files in this directory have this characterization status until a
file or field is backed by an explicitly recorded independent oracle.

## Reviewing and updating a fixture

Do not refresh a fixture merely to make a failing test pass.

1. Inspect the output diff and decide whether it exposes a defect or an
   intentional behavior change.
2. State the intended behavior from an independent source where one exists:
   an analytic result, domain invariant, specification, historical defect
   example, independently implemented reference calculation, or human-reviewed
   expected result.
3. If the implementation is wrong, fix it and retain the fixture. If the change
   is intentional, update only the affected values and record the rationale and
   oracle source in the change review.
4. When no independent oracle is available, an update may establish a new
   characterization baseline only after a human reviews the complete diff.
   Record that limitation explicitly; do not describe the new values as
   independently verified.
5. Run `uv run pytest tests/test_regression.py` and include the fixture diff in
   review.

## When to strengthen or replace this evidence

Replace or supplement the affected fixture fields with analytic/invariant
assertions or independently computed and reviewed expectations before they are
used as correctness evidence for a release or research claim. Also strengthen
them when an escaped defect shows that output equality missed an important
requirement, when reviewers cannot explain why a frozen value is correct, or
when repeated legitimate changes make the snapshot costly to review.

The strengthened test should name the requirement it protects and retain the
frozen fixture only if its broader change-detection value still justifies its
maintenance cost.
