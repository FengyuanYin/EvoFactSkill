"""Versioned optimizer guidance for feedback batch routes, never ordinary evolve."""

import hashlib

REGRESSION_POLICY_VERSION = "paired_regression_review_v1"
REGRESSION_REVIEW_POLICY = """
Paired regression review policy:
Before selecting targets or proposing edits, give current training errors and
historical source-validation regressions equal diagnostic priority. Use only
authorized supervision. Match baseline and candidate records by sample_id within
the SAME evaluated attempt, not across different validation sets. Separate:
1. repaired: baseline wrong, candidate correct;
2. regressed: baseline correct, candidate wrong;
3. preserved: both correct;
4. unresolved: both wrong.
Keep runtime failures separate; do not count them as judgment errors or repairs.
If labels or paired traces are unavailable, state the uncertainty rather than
inventing outcomes. Historical results describe that candidate, not an untested edit.

For each proposed behavioral rule, inspect supporting errors AND available correct
counterexamples with the same apparent cues, especially REAL news that a rejected
candidate turned into FAKE. Explain an observable distinction that could repair the
error while preserving these cases. Missing provenance, emotional wording, vivid
details and narrative coherence are not standalone veracity rules. For datasets
without external evidence, do not require external verification as a prerequisite
for REAL. Do not merely raise or lower the global skepticism threshold.

Define when the change applies, when the original behavior must remain, and which
responsible Agent can produce the missing discriminating analysis. Edit Router,
Specialist or Judge according to the diagnosed cause; multiple coordinated edits
remain allowed, with no quota. A rationale alone does not establish causal attribution.

Review prior rejected hypotheses and their REAL/FAKE regressions. Rewording a failed
rule is not new evidence. Revisit it only with a new supported distinction and explain
why the earlier counterexamples should be preserved. If evidence does not support a
safe generalizable edit, return no_change; do not invent a change to obtain promotion.
When source-validation supervision is absent, use available training/verified-challenge
successes and failures and report the missing counterexamples explicitly. Never access
independent holdout or final-test rows to satisfy this review.

Use only the existing JSON response schema. In target selection, summarize the
supported mechanism and regression risk in rationale. In the final proposal rationale
(and each change's rationale), cite supplied trace IDs for supporting errors and
counterexamples when available, describe the applicability boundary, and explain
how this differs from rejected hypotheses. Add no new JSON fields. Trace IDs and
case-specific labels belong in rationale only, never in reusable Agent instructions.
Preserve label contracts, role boundaries, security checks and promotion rules.
""".strip()


def regression_policy_identity():
    return {
        "version": REGRESSION_POLICY_VERSION,
        "prompt_digest": hashlib.sha256(REGRESSION_REVIEW_POLICY.encode()).hexdigest(),
    }


def append_regression_review(system, context):
    marker = context.get("optimizer_regression_policy")
    if marker is None:
        return system
    if not context.get("feedback_enabled") or marker != regression_policy_identity():
        raise ValueError("optimizer regression policy requires matching feedback mode")
    return system + "\n\n" + REGRESSION_REVIEW_POLICY
