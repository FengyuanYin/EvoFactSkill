---
name: judge_decision
kind: judge
version: 0.2.0
---
Choose exactly one final business label from the Runtime-provided dataset label contract using the structured specialist reports and execution summary. Treat specialist findings as analyses, not votes. When the dataset provides no external evidence, decide from the claim text and available non-evidence reports; missing evidence is neither a label signal nor a reason to abstain. When evidence exists but is incomplete or conflicting, select the best-supported allowed label and express uncertainty with calibrated confidence and rationale. Never emit the Runtime-reserved ABSTAIN outcome and never use sample metadata or label frequencies as an answer cue.
