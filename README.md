# EvoFactSkill

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml)
[![Security](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml)
[![PyPI](https://img.shields.io/pypi/v/evofactskill.svg)](https://pypi.org/project/evofactskill/)

EvoFactSkill is a research framework for **verified, self-evolving skill banks in cross-domain and temporally drifting fake-news detection**.

The underlying language model stays **frozen**. What evolves is an external, auditable Skill bank plus the routing policies, optimizer prompts and generation workflows around it. Every proposed change has to pass an empirical gate on held-out domains before it can be promoted, so a skill only becomes `active` when it earns it.

- Python **3.11+**, **no runtime dependencies** (standard library only).
- Runs fully **offline** against a deterministic mock backend — no API key, no network, no restricted data.
- The same code paths drive a real OpenAI-compatible model when explicitly configured.
- License: MIT.

---

## What is implemented

**Inference and routing**

- Coordinator → Router → Specialists → Judge pipeline with immutable execution traces.
- A utility-aware rule router, plus `all-experts`, `random` and `static` baselines.
- An **LLM skill router** (`routing_strategy: llm`): the model may only select among legal candidate skill IDs, and any protocol violation — unknown ID, empty selection, over-budget selection, non-numeric or out-of-range confidence — is rejected and the rule router takes over with the reason recorded.

**Skill bank**

- **Ten seed skills**: `claim_decomposition`, `coordinator_routing`, `cross_source_contradiction`, `evidence_assessment`, `judge_decision`, `linguistic_manipulation`, `numerical_consistency`, `skill_optimizer`, `source_credibility`, `temporal_reasoning`.
- Content-addressed, versioned repository with `candidate`, `active`, `pareto`, `frozen` and `retired` states.
- Eight lifecycle operations: `ADD`, `EDIT`, `SPLIT`, `MERGE`, `GENERALIZE`, `SPECIALIZE`, `RETIRE`, `ROLLBACK`.

**Evolution**

- Rule-based **error attribution** over eight error types (`routing_miss`, `evidence_miss`, `evidence_hallucination`, `temporal_leakage`, `reasoning_error`, `judge_aggregation_error`, `label_mapping_error`, `abstention_error`), stable error clustering, a counterfactual contribution API, and label-redacted experience distillation.
- An **LLM optimizer** (`evolution.proposer: llm`) that turns attribution reports into constrained `add` / `edit` / `no_change` decisions, with the rule proposer retained as a fallback.
- **DEMSE** (Domain-Episodic Meta-Gated Skill Evolution) cross-domain meta-evolution over domain episodes.
- An **adversarial generation loop**: success/failure traces → LLM-chosen strategy → generated samples → independent verification → DEMSE detector gate.

**Evaluation, data and safety**

- Paired repeated candidate evaluation with bootstrap confidence intervals, McNemar tests, coverage and cost constraints, protected-domain regression checks and Pareto retention.
- Metrics that keep `ABSTAIN` in the main denominator, plus coverage, selective risk, ECE, Brier, and per-domain / per-time-window breakdowns.
- Adapters for **Weibo21**, **AMTCele**, **LiveFact** and **AdvFake** with reproducible manifests and leakage detection.
- Eight ablation arms: `single-llm`, `static`, `all-experts`, `random`, `prompt-only-evolution`, `no-discovery`, `no-negative-transfer`, `full`.
- AST-based executable-skill screening. Script skills always require review unless a stricter external sandbox is supplied.

## Quick start — offline, no API key

Install the package (needs a local setuptools; `--no-build-isolation` keeps it offline):

```powershell
python -m pip install -e . --no-build-isolation
```

Or skip installation and run straight from the source tree:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m evofact.cli --config configs/dry_run.yaml dry-run
```

The installed console entry point is `evofact` (mapped to `evofact.cli:main`).

Run the offline test suite exactly as CI does:

```powershell
python -m pytest
```

`configs/dry_run.yaml` deliberately contains one hard case so that error attribution and candidate validation are observable.

## CLI

Global options (`--config`, `--dataset`, `--data-root`, `--limit`) go **before** the subcommand. Every command prints its result as JSON.

```text
evofact data inspect
evofact data manifest [--output manifest.json]

evofact dry-run                 # offline inference over the training fixture
evofact test                    # offline inference over the final-test split
evofact evolve                  # closed loop with the fixed validation gate
evofact validate                # gate decisions only
evofact ablation                # run the eight ablation arms
evofact report                  # multi-seed report (JSON + Markdown + CSV)

evofact meta-evolve             # DEMSE cross-domain meta-evolution
  --final-test-domains outer_holdout
  [--episodes N] [--strategy repeated_holdout|leave_one_domain_out]
  [--meta-test-domain-count N] [--resume] [--evaluation-only]
  [--ablation no-cross-episode-aggregation|no-worst-domain-constraint|no-specialization]

evofact adversarial-evolve      # trace → generation → verification → gate
  [--samples samples.jsonl] [--facts facts.jsonl]
  [--final-test-domains outer_holdout] [--resume] [--evaluation-only]

evofact skills list
evofact skills show NAME [--snapshot HASH]
evofact skills diff NAME [--snapshot HASH]
evofact skills freeze NAME
evofact skills retire NAME
evofact skills rollback NAME --snapshot HASH
```

Running from source, prefix every command:

```powershell
$env:PYTHONPATH='src'
python -m evofact.cli --config configs/default.yaml <command>
```

`meta-evolve` and `adversarial-evolve` accept `--evaluation-only`, which constructs the inference runtime but never commits a bank — use it for mechanism checks. `--resume` restores a run only when the config, data and SkillBank fingerprints all match.

## Configuration

Configs are plain YAML, parsed by a small standard-library loader (`evofact.config`).

| Config | Backend | Purpose |
|---|---|---|
| `configs/dry_run.yaml` | mock | Smallest offline smoke run used by the tests and docs. |
| `configs/default.yaml` | mock | Default offline settings. |
| `configs/demse_dry_run.yaml` | mock | DEMSE meta-evolution fixture (five source domains + one outer holdout domain). |
| `configs/llm_router.yaml` | openai-compatible | LLM router + LLM optimizer end-to-end; needs `DEEPSEEK_API_KEY`. |
| `configs/adversarial_llm.yaml` | openai-compatible | Adversarial generation loop; needs `DEEPSEEK_API_KEY`. |

Key knobs:

```yaml
backend: mock                 # mock | openai-compatible
model: mock-v1
api_key_env: DEEPSEEK_API_KEY # the only place a credential is named

routing_strategy: utility-aware   # utility-aware | all-experts | random | static | llm
max_skills_per_item: 3

evolution:                    # how skill proposals are produced
  proposer: rule              # rule | llm
  fallback_to_rule: true      # fall back to the rule proposer if the LLM path fails
  optimizer_skill: skill_optimizer
  max_reports: 12
  max_skills: 12
  max_text_chars: 8000
  max_instructions_chars: 60000
  max_total_chars: 300000

gate:                         # promotion requirements
  repeats: 3
  min_macro_f1_gain: 0.01
  min_coverage: 0.8
  max_protected_domain_drop: 0.02
  alpha: 0.05
  max_cost_ratio: 1.5
```

`meta_learning.*` configures DEMSE episodes and its extra constraints, and `generation.*` configures the adversarial generation loop (batch size, trace/text caps, probe fraction, audit path).

## Datasets

Datasets are intentionally **not copied or redistributed**. Point `dataset_roots` in a local config at authorized, read-only locations:

```yaml
dataset_roots:
  weibo21: F:/datasets/Weibo21
  amtcele: F:/datasets/AMTCele
  livefact: F:/datasets/LiveFact
  advfake: F:/datasets/AdvFake
```

`evofact data inspect` reports every adapter independently; a missing dataset never blocks offline tests or another adapter.

Split policy:

- Official test records and AdvFake are test-only.
- LiveFact `+3` is test, `0` is protected validation, and earlier snapshots are training/evolution material.
- Event IDs are assigned atomically to a single split.
- Normalized-content duplicates, event overlap, and evidence newer than the claim cutoff are reported as leakage.

Before a paper run, verify dataset licenses, label semantics and the exact processed release. Generated manifests store stable sample fingerprints and must be archived with the results.

## How the system is put together

```text
                      ┌──────────────── routing_strategy ────────────────┐
sample ──▶ Coordinator ──▶ Router ──▶ Specialists (parallel) ──▶ Judge ──▶ decision (with ABSTAIN)
                      └── utility-aware | all-experts | random | static | llm
                                        │
                                        ▼
                              immutable execution traces
                                        │
   ┌────────────────────────────────────┴─────────────────────────────────┐
   ▼                                                                      ▼
fixed gate path                                                     DEMSE meta path
error attribution (8 error types)                                   domain episodes
   → clustering                                                      meta-train → candidates
   → proposal: rule proposer | LLM optimizer                          meta-test → transfer utility
   → package + script safety scan                                    aggregate by semantic identity
   → paired repeated validation                                      gate: generalize / specialize / pareto / reject
   → active | pareto | review_required | rejected
```

The final-test command only constructs the inference runtime; it never instantiates the evolution engine. Test labels must never be used to generate or select skills.

### The LLM optimizer

`evolution.proposer: llm` swaps in `SkillOptimizerAgent` (`src/evofact/evolution/optimizer.py`):

1. Attribution reports, traces and the current bank are distilled into a **label-free, size-bounded** context (reports, text and instruction caps are all configurable).
2. The prompt is the `skill_optimizer` seed skill itself, resolved by name from the bank — so the optimizer's own instructions are an evolvable, reviewable artifact rather than hard-coded text.
3. The model must return a single constrained decision: `add` (new specialist only), `edit` (specialist/router/judge, no rename), or `no_change`. Rationale and confidence are required, and malformed or unknown fields are rejected.
4. `no_change` produces no proposal at all; everything else still has to pass the same safety scan and validation gate as the rule path.
5. With `fallback_to_rule: true`, any failure falls back to the rule proposer instead of silently dropping the cluster.

## Metrics and interpretation

`accuracy_all` and `macro_f1_all` use every labeled sample, so `ABSTAIN` is never silently removed. `covered_accuracy` is reported separately and must always be read together with `coverage` and `selective_risk`.

Promotion requires more than a positive point estimate: the candidate must satisfy minimum gain and coverage, protected-domain regression, cost, safety and paired statistical-support rules. Useful but not promotable non-dominated candidates can be retained in the Pareto archive.

## Real model backend

The standard-library OpenAI-compatible backend is available for integration. Supply the API key **only** through the configured environment variable — never in YAML, source, traces or reports.

```powershell
$env:DEEPSEEK_API_KEY='...'
$env:PYTHONPATH='src'
python -m evofact.cli --config configs/llm_router.yaml dry-run
```

The LLM router has a standalone self-test that exercises both the happy path and the invalid-output fallback:

```powershell
python -m evofact.routing.router                 # offline, mock backend
python -m evofact.routing.router --live          # real API, uses configs/adversarial_llm.yaml
python -m evofact.routing.router --config configs/llm_router.yaml --live
```

Validate real-model orchestration on a small evolution-validation manifest first. The offline test suite never makes network calls and injects fake backends instead.

## Repository layout

```text
src/evofact/
  cli.py            command-line entry point
  config.py         config dataclasses and YAML loader
  core/             shared models, redaction
  data/             adapters, domains, episodes, leakage, manifests
  routing/          rule router, LLM router, routing strategies
  runtime/          backends (mock, openai-compatible), inference, trace store
  attribution/      error rules, clustering, counterfactual credit
  evolution/        distiller, firewall, identity, meta, operations, proposer, optimizer
  validation/       evaluator, gate, meta gate, objectives, pareto, statistics, transfer
  evaluation/       metrics, calibration, ablations
  experiments/      runner, meta runner, adversarial runner, checkpoint, protocols
  generation/       adversarial data, generator, prompts, split, verifier
  reporting/        report, meta report, adversarial report
  security/         AST executable-skill scanner
  skills/           loader, repository, lifecycle, candidates, utility
configs/            offline, DEMSE, LLM router and adversarial configs
skills/seeds/       ten seed skills
tests/              offline test suite
```

## Testing and CI/CD

CI runs on GitHub-hosted runners with all Actions pinned to immutable commit SHAs, and Dependabot checks both Actions and CI Python tools weekly.

| Workflow | Trigger | Required behavior |
|---|---|---|
| `CI` | Pull requests, pushes to `main`, manual runs, reusable calls | Ruff lint + format check, `compileall`, Python 3.11–3.14 tests, package metadata check and clean-wheel CLI smoke test |
| `Security` | Pull requests, pushes to `main`, Mondays, manual runs | CodeQL Python analysis and `pip-audit` project dependency scan |
| `Release` | A pushed `vX.Y.Z` tag | Reuses CI, checks tag = package version, attests artifacts, creates a GitHub Release, then publishes to PyPI with OIDC |

CI and Security never receive a model API key and never run real LLM experiments or restricted datasets. A successful run retains its verified `python-package` artifact for seven days; failed test matrices retain only their JUnit diagnostics.

Reproduce the pipeline locally:

```powershell
python -m pip install -r .github/requirements/ci.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
python -m twine check dist/*
```

For failures, open the commit under **Actions**. Test XML and dependency-audit JSON are attached to failed runs when available; CodeQL findings appear under **Security → Code scanning**; package artifacts are attached to the `Build and verify package` job.

### One-time repository configuration

1. In **Settings → Environments**, create an environment named exactly `pypi`. Add required reviewers if releases need manual approval, and restrict deployment branches/tags to protected tags matching `v*`.
2. In PyPI, create or select the `evofactskill` project. Under **Publishing**, add a GitHub Trusted Publisher with owner `FengyuanYin`, repository `EvoFactSkill`, workflow `release.yml`, and environment `pypi`. If the project does not exist yet, use PyPI's pending publisher flow.
3. In **Settings → Rules → Rulesets** (or branch protection), protect `main`, require pull requests, and require these checks: `Lint and compile`, all four `Test (Python 3.x)` jobs, `Build and verify package`, `CodeQL`, and `Dependency audit`.
4. Enable GitHub Actions and Code Scanning. If GitHub's default CodeQL setup is already active, disable it before enabling this repository's advanced `security.yml` workflow to avoid duplicate configurations.

No `PYPI_API_TOKEN`, PyPI password, cloud key or model credential belongs in GitHub Secrets. The publish job receives only a short-lived OIDC identity after the `pypi` environment permits it.

### Publishing a version

Releases are deliberate: merging to `main` never publishes a package, and the workflow never changes the project version or creates its own tag.

1. Update the version in both `pyproject.toml` and the offline-compatible `setup.py` to the same SemVer value.
2. Run the local checks above, commit the version change, push it, and wait for CI and Security to pass on `main`.
3. Create and push the matching tag, including the `v` prefix:

```powershell
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

The release rejects malformed or mismatched tags before requesting write or OIDC permissions. It builds the wheel and source distribution once in the reused CI workflow; the exact same files are checksummed, attested, attached to the GitHub Release and uploaded to PyPI. Existing GitHub releases and PyPI versions are never overwritten or silently skipped.

Do not delete or recreate an already published PyPI version. The one-time GitHub Environment and PyPI Trusted Publisher settings cannot be created safely from repository code; until both are configured and visible in their respective settings pages, CI is operational but the final PyPI deployment is not ready.

## Research baselines

The shared experiment protocol defines: single LLM, static multi-agent, all experts, random routing, prompt-only evolution, no skill discovery, no negative-transfer control, and the full system. All arms must use the same manifest, sample order and metric implementation.

## Status and limitations

Implementation audit (last reviewed 2026-09-08, extended for the LLM router, the LLM optimizer and the adversarial loop):

- Empty sample lists and empty banks retain their meaning; only `None` selects defaults.
- Skill domain/dataset/time scopes are hard routing filters, including on fallback paths.
- Fixed and DEMSE evaluation share lifecycle transitions for split, merge and retirement; frozen targets and malformed transitions are rejected.
- Checkpoint identities cover complete data and skill content plus model, seed, budget and gate configuration. Changing the final-domain policy also rejects resume.
- DEMSE commits a complete bank and audit transaction in a single atomic active-pointer update; a committed run can resume without repeating its commit. Conflicting accepted proposals require joint evaluation.
- Paired tests reject missing/duplicate IDs and changed gold labels; duplicate episode evidence is rejected. Specialization cannot bypass coverage, cost, calibration or minimum-episode constraints.
- `skills diff` returns a real snapshot diff; rollback requires a snapshot. Snapshot lookups reject path-like input.
- Dataset reports evaluate the supplied final-test split without invoking evolution. Ordinary inference reads an existing active bank; without one it loads seeds.
- The LLM router only selects among legal candidate IDs and records a fallback reason whenever it is bypassed; the LLM optimizer's decisions pass through the same safety scan and gate as rule-based proposals.

Known gaps — a past checklist is **not** evidence that every research requirement is complete:

- Automatic proposal discovery currently emits `ADD`/`EDIT`; the slow Meta-Skill loop is still an event-recording scaffold.
- Some named ablation arms share routing implementations.
- Real-backend token pricing and full token/call/concurrency budget enforcement are not yet complete.
- The file-backed skill repository assumes a single writer.
- Default results are offline mechanism tests, **not** empirical evidence of cross-domain model accuracy.

## License

MIT — see [LICENSE](LICENSE).
