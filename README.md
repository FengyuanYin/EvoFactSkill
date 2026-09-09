# EvoFactSkill

[![CI](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/ci.yml)
[![Security](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/FengyuanYin/EvoFactSkill/actions/workflows/security.yml)
[![PyPI](https://img.shields.io/pypi/v/evofactskill.svg)](https://pypi.org/project/evofactskill/)

## GitHub CI/CD and trusted releases

The repository uses three least-privilege GitHub Actions workflows. All referenced Actions are pinned to immutable commit SHAs and Dependabot checks both Actions and CI Python tools every week.

| Workflow | Trigger | Required behavior |
|---|---|---|
| `CI` | Pull requests, pushes to `main`, manual runs, reusable calls | Ruff lint/format, Python 3.11–3.14 tests, package metadata and clean-wheel CLI smoke test |
| `Security` | Pull requests, pushes to `main`, Mondays, manual runs | CodeQL Python analysis and `pip-audit` project dependency scan |
| `Release` | A pushed `vX.Y.Z` tag | Reuses CI, checks tag = package version, attests artifacts, creates a GitHub Release, then publishes to PyPI with OIDC |

CI and Security never receive a model API key and do not run real LLM experiments or restricted datasets. A successful CI run retains its verified `python-package` artifact for seven days. Failed test matrices retain only their JUnit diagnostic files.

### One-time repository configuration

1. In **Settings → Environments**, create an environment named exactly `pypi`. Add required reviewers if releases need manual approval, and restrict deployment branches/tags to protected tags matching `v*`.
2. In PyPI, create or select the `evofactskill` project. Under **Publishing**, add a GitHub Trusted Publisher with owner `FengyuanYin`, repository `EvoFactSkill`, workflow `release.yml`, and environment `pypi`. If the project does not exist yet, use PyPI's pending publisher flow.
3. In **Settings → Rules → Rulesets** (or branch protection), protect `main`, require pull requests, and require these checks: `Lint and compile`, all four `Test (Python 3.x)` jobs, `Build and verify package`, `CodeQL`, and `Dependency audit`.
4. Enable GitHub Actions and Code Scanning. If GitHub's default CodeQL setup is already active, disable it before enabling this repository's advanced `security.yml` workflow to avoid duplicate configurations.

No `PYPI_API_TOKEN`, PyPI password, cloud key, or model credential belongs in GitHub Secrets. The publish job receives only a short-lived OIDC identity after the `pypi` environment permits it.

### Publishing a version

Releases are deliberate: merging to `main` never publishes a package, and the workflow never changes the project version or creates its own tag.

1. Update the version in both `pyproject.toml` and the offline-compatible `setup.py` to the same SemVer value.
2. Run the local checks below, commit the version change, push it, and wait for CI and Security to pass on `main`.
3. Create and push the matching tag, including the `v` prefix:

```powershell
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

The release rejects malformed or mismatched tags before requesting write or OIDC permissions. It builds the wheel and source distribution once in the reused CI workflow; the exact same files are checksummed, attested, attached to GitHub Release, and uploaded to PyPI. Existing GitHub releases and PyPI versions are never overwritten or silently skipped.

### Local release checks

Run these commands in a disposable virtual environment before tagging:

```powershell
python -m pip install -r .github/requirements/ci.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
python -m twine check dist/*
```

For failures, open the commit under the repository's **Actions** tab. Test XML and dependency-audit JSON are attached to the failed run when available; CodeQL findings appear under **Security → Code scanning**. Package artifacts are attached to the `Build and verify package` job. After fixing code or an external GitHub/PyPI setting, push a new commit or use **Re-run failed jobs** for the unchanged release tag. Do not delete or recreate an already published PyPI version.

The one-time GitHub Environment and PyPI Trusted Publisher settings cannot be created safely from repository code. Until both are configured and visible in their respective settings pages, CI is operational but the final PyPI deployment is not ready.

新增独立链路：**meta-train 成功/失败 trace → LLM 自选策略并一次生成 samples → 独立审核 → DEMSE 检测技能门控**。成功类型升难度，失败类型增样强化；底层模型参数冻结。

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m evofact.cli --config configs/adversarial_llm.yaml adversarial-evolve --evaluation-only
python -m evofact.cli --config configs/adversarial_llm.yaml adversarial-evolve --evaluation-only --resume
```

该命令仅支持 `proposer: llm`，运行前配置 `DEEPSEEK_API_KEY`；即使不传输入而使用模拟夹具，也会调用真实 LLM。每个 episode 一次生成调用，同时输出 samples 和策略 decisions；独立标签审核另有一次调用。`--samples samples.jsonl` 读取带 evidence 的标准化输入，`--facts` 可选。去掉 evaluation-only 才提交检测库与正式生成审计；样本按 episode 导出 JSONL。旧三模式和数值模板已删除。详见 [设计与输入协议](docs/adversarial-design.md) 和 [Day 10 教程](learn/Day10-挑战生成智能体与双层进化.md)。

EvoFactSkill 以 DEMSE（Domain-Episodic Meta-Gated Skill Evolution，域情景元门控技能进化）作为跨域自进化核心：每个 episode 将完整源领域划分为 meta-train 与 meta-test，前者生成可解释 Skill 候选，后者模拟未知域并只提供迁移效用门控。多轮结果按候选语义身份聚合后，系统再决定泛化、特化、Pareto 保留或拒绝。底层语言模型始终冻结，因此这里的“元学习”发生在离散 SkillBank 空间，而不是参数级 MAML。

离线机制验证：

```powershell
python -m evofact.cli --config configs/demse_dry_run.yaml meta-evolve --final-test-domains outer_holdout --evaluation-only
```

可用 `--episodes`、`--strategy repeated_holdout|leave_one_domain_out` 与 `--meta-test-domain-count` 覆盖 episode 计划；`--resume` 仅在配置、数据和 SkillBank 指纹均一致时恢复。

论文消融通过 `--ablation no-cross-episode-aggregation`、`--ablation no-worst-domain-constraint` 或 `--ablation no-specialization` 运行；原 `evolve` 命令提供固定验证门控对照。

该命令使用五个源领域和一个外层留出领域的确定性 MockBackend 夹具，输出 JSON、Markdown 与 CSV 报告。Mock 数值仅用于验证域隔离、候选生成、迁移聚合和门控流程，不代表真实模型实验结果。正式实验应通过 `--dataset`、`--data-root` 和显式 `--final-test-domains` 指定真实数据，并确保至少三个源领域。

EvoFactSkill is a research framework for **verified, self-evolving skill banks in cross-domain and temporally drifting fake-news detection**. It keeps the underlying language model frozen and evolves auditable external skills, routing policies and workflows from execution traces.

The repository is independent from CD-FND. It can read separately configured dataset locations but never imports or writes the original project.

## What is implemented

- Coordinator → utility-aware Skill Router → Specialists → Judge inference with immutable traces.
- Nine seed skills and a versioned, content-addressed Skill repository.
- `ADD`, `EDIT`, `SPLIT`, `MERGE`, `GENERALIZE`, `SPECIALIZE`, `RETIRE`, and `ROLLBACK` lifecycle operations.
- Rule-based error attribution, stable error clustering, counterfactual contribution API, experience distillation and structured proposals.
- Paired repeated candidate evaluation, bootstrap confidence intervals, McNemar test, coverage/cost constraints, protected-domain regression checks and Pareto outcomes.
- Metrics that count abstention in the main denominator, plus coverage, selective risk, ECE, Brier, domain/time metrics and cost.
- Adapters for Weibo21, AMTCele, LiveFact and AdvFake with reproducible manifests and leakage checks.
- Eight common ablation arms and deterministic offline execution without an API key.
- AST-based executable-skill screening. Script skills always require review unless a stricter external sandbox is supplied.

## Quick verification — no network or API key

安装到本地环境：`python -m pip install -e . --no-build-isolation`（需要本地 setuptools）。控制台入口已统一到 `evofact.cli:main`；无需安装也可按下方 `PYTHONPATH` 方式运行。

中文源码教程见 [learn/README.md](learn/README.md)：九天内容覆盖入口、数据契约、隔离、推理、归因、统计、生命周期、DEMSE 与验收，每章都有实际代码对应的练习和答案。

PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
python -m unittest discover -s tests -p 'test*.py' -v
python -m evofact.cli --config configs/dry_run.yaml dry-run
python -m evofact.cli --config configs/dry_run.yaml validate
python -m evofact.cli --config configs/dry_run.yaml ablation
```

An offline source build can be checked with `python setup.py build`. Building a wheel additionally requires the standard `wheel` build plugin.

The dry-run deliberately contains one hard case so error attribution and candidate validation are observable.

## CLI

```text
evofact data inspect
evofact data manifest [--output manifest.json]
evofact dry-run
evofact evolve
evofact validate
evofact test
evofact ablation
evofact report
evofact skills list
evofact skills show NAME [--snapshot HASH]
evofact skills diff NAME
evofact skills freeze NAME
evofact skills retire NAME
evofact skills rollback NAME --snapshot HASH
```

When running from source, prefix commands with:

```powershell
$env:PYTHONPATH='src'
python -m evofact.cli --config configs/default.yaml
```

## Dataset setup

Datasets are intentionally not copied or redistributed. Point `dataset_roots` in a local config at authorized, read-only locations:

```yaml
dataset_roots:
  weibo21: F:/datasets/Weibo21
  amtcele: F:/datasets/AMTCele
  livefact: F:/datasets/LiveFact
  advfake: F:/datasets/AdvFake
```

`evofact data inspect` reports every adapter independently. Missing datasets do not prevent offline tests or use of another adapter.

Split policy:

- Official test records and AdvFake are test-only.
- LiveFact `+3` is test, `0` is protected validation, and earlier snapshots are training/evolution material.
- Event IDs are assigned atomically to a split.
- Normalized-content duplicates, event overlap, and evidence newer than the claim cutoff are reported as leakage.

Before a paper run, verify dataset licenses, label semantics and the exact processed release. Generated manifests store stable sample fingerprints and must be archived with results.

## Evolution protocol

```text
training traces
  → error attribution and counterfactual credit
  → compact, label-redacted experience distillation
  → lifecycle proposal
  → package and script safety scan
  → paired repeated validation
  → performance/coverage/calibration/cost/domain constraints
  → active | pareto | review_required | rejected
```

The final-test command only constructs the inference runtime. It does not instantiate the evolution engine. Test labels must never be used to generate or select skills.

## Metrics and interpretation

`accuracy_all` and `macro_f1_all` use every labeled sample. `ABSTAIN` is therefore not silently removed. `covered_accuracy` is reported separately and must always be read together with `coverage` and `selective_risk`.

Promotion requires more than a positive point estimate. The candidate must satisfy minimum gain and coverage, protected-domain regression, cost, safety, and paired statistical-support rules. Useful but not promotable non-dominated candidates can be retained in the Pareto archive.

## Real model backend

The standard-library OpenAI-compatible backend is available for integration. Supply the API key only through the configured environment variable. Never place credentials in YAML, source code, traces, or reports. Real-model orchestration should first be validated on a small evolution-validation manifest; the offline tests never make network calls.

## Research baselines

The shared experiment protocol defines: single LLM, static multi-agent, all experts, random routing, prompt-only evolution, no skill discovery, no negative-transfer control, and the full system. All arms must use the same manifest, sample order and metric implementation.

See [spec.md](spec.md), [plan.md](plan.md), [task.md](task.md), and [checklist.md](checklist.md) for the approved scope and verification criteria.

## 2026-09-08 implementation audit

- Empty sample lists and empty banks retain their meaning; only `None` selects defaults.
- Skill domain/dataset/time scopes are hard routing filters, including fallback paths.
- Fixed and DEMSE evaluation share lifecycle transitions for split, merge and retirement; frozen targets and malformed transitions are rejected.
- Checkpoint identities cover complete data and skill content plus model, seed, budget and gate configuration. Changing the final-domain policy also rejects resume.
- DEMSE commits a complete bank and audit transaction in one atomic active-pointer update. A committed run can resume without repeating its commit. Conflicting accepted proposals require joint evaluation.
- Paired tests reject missing/duplicate IDs and changed gold labels; duplicate episode evidence is rejected. Specialization cannot bypass coverage, cost, calibration or minimum-episode constraints.
- `skills diff` returns a real snapshot diff; rollback requires a snapshot. Snapshot lookups reject path-like input.
- Dataset reports evaluate the supplied final-test split without invoking evolution. Ordinary inference reads an existing active bank; without one it loads seeds.

The historical checklist is not evidence that every research requirement is complete. Automatic proposal discovery currently emits ADD/EDIT; the slow Meta-Skill loop is an event-recording scaffold, and some named ablation arms share routing implementations. Real-backend token pricing and full token/call/concurrency budget enforcement are not yet complete. The current file repository assumes a single writer. Default results remain offline mechanism tests, not empirical evidence of cross-domain model accuracy.
