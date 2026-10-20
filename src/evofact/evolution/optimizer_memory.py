"""Persistent memory and token-aware requests for the three feedback batch routes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict

from evofact.core.budget_models import BudgetRequest, UsageDetails
from evofact.evolution.firewall import validate_generation_request
from evofact.runtime.backend import BackendResult, ModelOutputTruncated
from evofact.runtime.node_runner import _usage_details
from evofact.runtime.openai_backend import BackendHTTPError

from .optimizer_context import _sanitize


class OptimizerContextFailure(ValueError):
    """The protected request cannot fit, or the model window is unknown."""


class OptimizerProtocolFailure(ValueError):
    """One model proposal/selection is invalid; the batch may try another proposal."""


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def safe(value):
    return _sanitize(value, max_text_chars=0)  # Sanitization no longer clips strings.


def remember(state, audit, *, pending=None, training=()):
    """Save full consumed diagnostics; only the structured view is exposed by default."""
    records = state.setdefault("optimizer_memories", [])
    key = f"b{audit['batch']}-a{audit['attempt']}"
    if any(r["id"] == key for r in records):
        return
    proposal = audit.get("proposal", {})
    decision = audit.get("decision", {})
    summary = {
        "id": key,
        "batch": audit["batch"],
        "attempt": audit["attempt"],
        "target": (pending or {}).get("base_id"),
        "hypothesis": proposal.get("rationale", audit.get("reason", "")),
        "status": decision.get("disposition", audit.get("status")),
        "accepted": decision.get("accepted", False),
        "reason": decision.get("reason", audit.get("reason", "")),
        "failures": decision.get("regression_failures", []),
        "domains": audit.get("train_domains", []),
        "historical_observation": True,
    }
    if pending:
        package = pending["package"]
        summary["skill_name"] = package["manifest"]["name"]
        summary["candidate_digest"] = package["package_digest"]
        summary["candidate_version"] = package["manifest"]["version"]
        base = next(
            (p for p in state.get("packages", []) if p["skill_id"] == summary["target"]), None
        )
        if base:
            summary["base_digest"] = base["package_digest"]
            summary["base_version"] = base["manifest"]["version"]
        if "packages" in pending:
            summary["targets"] = pending["base_ids"]
            summary["hypothesis"] = pending.get("joint_rationale") or summary["hypothesis"]
            summary["joint_outcome"] = True
            summary["changes"] = [
                {
                    "skill_name": p["manifest"]["name"],
                    "candidate_digest": p["package_digest"],
                    "candidate_version": p["manifest"]["version"],
                    "base": next(
                        (
                            {
                                "skill_id": b["skill_id"],
                                "digest": b["package_digest"],
                                "version": b["manifest"]["version"],
                            }
                            for b in state.get("packages", [])
                            if b["manifest"]["name"] == p["manifest"]["name"]
                        ),
                        None,
                    ),
                }
                for p in pending["packages"]
            ]
        for role in ("baseline", "candidate"):
            result = pending[role]["result"]
            metrics = result.get("aggregate_metrics", {})
            summary[role] = {
                k: metrics[k]
                for k in ("n", "accuracy", "accuracy_all", "macro_f1_all")
                if k in metrics
            }
    # Independent validation contributes its refusal reason only, never its rows/metrics.
    records.append(
        {
            "id": key,
            "summary": safe(summary),
            "training_traces": plain(training),
            "baseline_traces": (pending or {}).get("baseline", {}).get("traces", []),
            "candidate_traces": (pending or {}).get("candidate", {}).get("traces", []),
            "proposal": proposal,
            "proposals": audit.get("proposals", [proposal]),
        }
    )


def _words(value):
    return set(re.findall(r"[\w-]+", json.dumps(value, ensure_ascii=False).casefold()))


def memory_view(records, context, forbidden):
    """Group repeated observations; relevant groups precede recent unrelated ones."""
    groups = defaultdict(list)
    query = _words(context.get("traces", []))
    for record in records:
        summary = record["summary"]
        try:
            validate_generation_request(summary, forbidden)
        except ValueError:
            continue
        groups[
            (
                tuple(summary.get("targets", [summary.get("target")])),
                tuple(c["skill_name"] for c in summary.get("changes", [])),
                tuple(summary.get("failures", [])),
                summary["status"],
            )
        ].append(record)
    ranked = []
    for items in groups.values():
        last = items[-1]["summary"]
        summary = dict(last, observation_count=len(items), source_ids=[r["id"] for r in items])
        score = len(
            query
            & _words({k: summary.get(k) for k in ("target", "skill_name", "domains", "reason")})
        )
        ranked.append((score, len(items), last["batch"], last["attempt"], summary, items[-1]))
    ranked.sort(key=lambda r: r[:4], reverse=True)
    examples = []
    for score, _, _, _, _, record in ranked:
        if score == 0:
            continue
        for trace in record["training_traces"] + record["candidate_traces"]:
            trace = safe(trace)
            try:
                validate_generation_request(trace, forbidden)
            except ValueError:
                continue
            examples.append({"memory_id": record["id"], "trace": trace})
            break
    return {
        "summary_version": 1,
        "observations": [r[4] for r in ranked],
        "retrieved_traces": examples,
    }


def _unique(rows):
    seen, result = set(), []
    for row in rows:
        key = digest(row)
        if key not in seen:
            result.append(row)
            seen.add(key)
    return result


def reduce_context(context):
    """Remove one optional unit, retaining whole strings and the current target package."""
    memory = context.get("training_memory", {})
    if memory.get("retrieved_traces"):
        return {"kind": "old_trace", "item": memory["retrieved_traces"].pop()["memory_id"]}
    feedback = context.get("validation_feedback", [])
    if len(feedback) > 1:
        feedback.pop()
        return {"kind": "older_feedback", "summary_retained_in_memory": True}
    if memory.get("observations"):
        return {"kind": "old_memory", "item": memory["observations"].pop()["source_ids"]}
    if context.get("prior_proposals"):
        context["prior_proposals"].pop()
        return {"kind": "old_proposal"}
    if feedback:
        latest = feedback[0]
        packages = [latest.get("rejected_package", {})] + latest.get("rejected_packages", [])
        for package in packages:
            if any(
                "content" in item or "content_base64" in item for item in package.get("files", [])
            ):
                package["files"] = [
                    {k: v for k, v in item.items() if k not in {"content", "content_base64"}}
                    for item in package["files"]
                ]
                return {"kind": "rejected_package_file_contents", "audit_preserved": True}
        groups = [latest.get("baseline_traces", []), latest.get("candidate_traces", [])]
        group = max(groups, key=len)
        if len(group) > 1:
            group.pop()
            return {"kind": "redundant_validation_trace"}
    if len(context.get("traces", [])) > 1:
        context["traces"].pop()
        return {"kind": "training_trace"}
    return None


def is_overflow(exc):
    if not isinstance(exc, BackendHTTPError) or not re.search(r"HTTP (400|413|422)\b", str(exc)):
        return False
    return any(
        term in str(exc).casefold()
        for term in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "context length",
            "too many tokens",
            "max context",
            "上下文长度",
        )
    )


def failure_kind(exc):
    if is_overflow(exc):
        return "context_overflow"
    if isinstance(exc, ModelOutputTruncated):
        return "output_truncated"
    if isinstance(exc, OptimizerProtocolFailure):
        return "proposal_protocol_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, BackendHTTPError):
        match = re.search(r"HTTP (\d+)", str(exc))
        return "http_" + match[1] if match else "http_error"
    if isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    return "runtime_failure"


class ManagedOptimizerBackend:
    """Wrap only batch-route optimizer calls; normal evolve never creates this adapter."""

    def __init__(
        self, backend, policy, state, save, *, budget=None, forbidden=(), component="optimizer"
    ):
        self.component = component
        self.backend = copy.copy(backend)
        self.configured_output_cap = getattr(backend, "max_output_tokens", None)
        self.backend._managed_optimizer = True
        self.policy, self.state, self.save = policy, state, save
        self.budget, self.forbidden = budget, forbidden
        self.encoder = None
        if policy.tokenizer_encoding:
            try:
                import tiktoken

                self.encoder = tiktoken.get_encoding(policy.tokenizer_encoding)
            except (ImportError, ValueError):
                pass

    def tokens(self, system, context):
        text = system + json.dumps(context, ensure_ascii=False, default=str)
        count = len(self.encoder.encode(text)) if self.encoder else len(text.encode("utf-8"))
        factor = self.state.get(f"{self.component}_token_factor", 1.0)
        return math.ceil((count + 64) * factor)

    def system(self, context, skill, selection=False):
        name = (
            "optimizer_joint_selection_system"
            if selection
            and context.get("joint_optimization")
            and hasattr(self.backend, "optimizer_joint_selection_system")
            else "optimizer_selection_system"
            if selection
            else "optimizer_system"
        )
        method = getattr(self.backend, name, None)
        return (
            method(skill)
            if selection and method
            else method(context, skill)
            if method
            else skill.instructions
        )

    async def capabilities(self):
        identity = {
            "model": getattr(self.backend, "model", type(self.backend).__name__),
            "url": getattr(self.backend, "url", None),
        }
        cached = self.state.get(f"{self.component}_capabilities")
        if cached and cached["identity"] != identity:
            raise OptimizerContextFailure(f"{self.component} endpoint/model changed")
        cached_window = (cached or {}).get("value", {}).get("context_window")
        if cached and (
            self.policy.context_window is not None
            or (type(cached_window) is int and cached_window > 0)
        ):
            capabilities = cached["value"]
        else:
            capabilities = {}
            if self.policy.context_window is None:
                method = getattr(self.backend, "optimizer_model_capabilities", None)
                if method:
                    try:
                        capabilities = await method()
                    except Exception:
                        capabilities = {}
            self.state[f"{self.component}_capabilities"] = {
                "identity": identity,
                "value": capabilities,
            }
            self.save()
        window = self.policy.context_window or capabilities.get("context_window")
        if type(window) is not int or window <= 0:
            raise OptimizerContextFailure(
                f"model window unknown; configure {self.component}_memory.context_window"
            )
        output = (
            getattr(self.backend, "max_output_tokens", None) or self.policy.reserved_output_tokens
        )
        maximum = capabilities.get("max_output_tokens")
        if type(maximum) is int and maximum > 0:
            output = min(output, maximum)
        self.backend.max_output_tokens = output
        available = math.floor(window * (1 - self.policy.safety_fraction)) - output
        if available <= 0:
            raise OptimizerContextFailure(f"output reserve leaves no {self.component} input budget")
        if self.state.pop(f"{self.component}_context_error", None) is not None:
            self.save()
        return window, output, available

    def reduce(self, context):
        return reduce_context(context)

    def calibrate(self, usage, estimated):
        if usage.input_tokens > estimated:
            key = f"{self.component}_token_factor"
            old = self.state.get(key, 1)
            self.state[key] = max(old, old * usage.input_tokens / estimated * 1.1)

    def grow_output(self, request, output):
        """Grow only the managed call's output allowance, respecting explicit caps."""
        maximum = (
            self.state.get(f"{self.component}_capabilities", {})
            .get("value", {})
            .get("max_output_tokens")
        )
        window = request["window"]
        ceiling = math.floor(window * (1 - self.policy.safety_fraction)) - 1
        for cap in (self.configured_output_cap, maximum):
            if type(cap) is int and cap > 0:
                ceiling = min(ceiling, cap)
        return min(output * 2, ceiling)

    async def invoke(self, request, slot, payload, skill, *, selection, limit, output):
        stage = request.setdefault(slot, {"calls": []})
        if "response" in stage:
            return stage["response"]
        # Resume the saved effective payload after interruption, not a new selection.
        payload = stage.setdefault("payload", plain(payload))
        limit = stage.get("input_budget", limit)
        output = stage.get("output_reserve", output)
        overflows = sum(
            c["status"] in {"context_overflow", "output_truncated"} for c in stage["calls"]
        )
        if overflows > self.policy.max_overflow_retries:
            if stage["calls"][-1]["status"] == "output_truncated":
                raise ModelOutputTruncated("managed output retry budget exhausted")
            raise OptimizerContextFailure("provider context retry budget exhausted")
        # Older checkpoints saved the failed call before any larger cap was planned.
        if stage["calls"] and stage["calls"][-1]["status"] == "output_truncated":
            previous = stage["calls"][-1]["output_reserve"]
            if output <= previous:
                grown = self.grow_output(request, previous)
                if grown <= previous:
                    raise ModelOutputTruncated("managed output reached explicit or provider cap")
                limit -= grown - output
                output = grown
                stage.update(output_reserve=output, input_budget=limit)
                self.save()
        for retry in range(overflows, self.policy.max_overflow_retries + 1):
            self.backend.max_output_tokens = output
            system = self.system(payload, skill, selection)
            omitted = stage.setdefault("omitted", [])
            while self.tokens(system, payload) > limit:
                removed = self.reduce(payload)
                if removed is None:
                    if stage["calls"] and stage["calls"][-1]["status"] == "output_truncated":
                        raise ModelOutputTruncated("output expansion cannot fit protected input")
                    raise OptimizerContextFailure(
                        f"protected {self.component} input does not fit model window"
                    )
                omitted.append(removed)
            event = {
                "retry": retry,
                "call_attempt": len(stage["calls"]) + 1,
                "estimated_input_tokens": self.tokens(system, payload),
                "input_budget": limit,
                "output_reserve": output,
                "token_counter": self.policy.tokenizer_encoding
                if self.encoder
                else "conservative_utf8_bytes",
                "system_digest": digest(system),
                "payload_digest": digest(payload),
                "status": "inflight",
            }
            stage["inflight"] = event
            self.save()
            reservation = None
            called = False
            try:
                if self.budget:
                    unit = f"{self.component}-memory:{request['id']}:{slot}:{event['call_attempt']}"
                    async with self.budget.concurrency(unit):
                        reservation = await self.budget.reserve(
                            unit,
                            BudgetRequest(
                                calls=1,
                                tokens=event["estimated_input_tokens"] + output,
                                purpose=f"{self.component}_memory",
                            ),
                        )
                        called = True
                        result = await asyncio.wait_for(
                            self._call(payload, skill, selection),
                            self.budget.limits.call_timeout_ms / 1000,
                        )
                else:
                    called = True
                    result = await self._call(payload, skill, selection)
                usage = _usage_details(result)
                if reservation:
                    await self.budget.reconcile(reservation, usage)
                    reservation = None
                event.update(status="completed", usage=plain(asdict(usage)))
                self.calibrate(usage, event["estimated_input_tokens"])
                stage["response"] = plain(result.value)
                stage["calls"].append(event)
                stage.pop("inflight", None)
                self.save()
                return stage["response"]
            except Exception as exc:
                event.update(
                    status="context_overflow"
                    if is_overflow(exc)
                    else "output_truncated"
                    if isinstance(exc, ModelOutputTruncated)
                    else "runtime_failure",
                    error_type=type(exc).__name__,
                )
                event["failure_kind"] = failure_kind(exc)
                stage["calls"].append(event)
                stage.pop("inflight", None)
                usage = getattr(exc, "usage_details", None) or UsageDetails(calls=int(called))
                event["usage"] = plain(asdict(usage))
                self.calibrate(usage, event["estimated_input_tokens"])
                if reservation:
                    await self.budget.reconcile(reservation, usage)
                    reservation = None
                self.save()
                if isinstance(exc, ModelOutputTruncated):
                    grown = self.grow_output(request, output)
                    if retry == self.policy.max_overflow_retries or grown <= output:
                        raise
                    limit -= grown - output
                    output = grown
                    stage.update(output_reserve=output, input_budget=limit)
                    self.save()
                    continue
                if not is_overflow(exc) or retry == self.policy.max_overflow_retries:
                    if is_overflow(exc):
                        raise OptimizerContextFailure(
                            "provider context overflow after bounded retries"
                        ) from exc
                    raise
                removed = self.reduce(payload)
                if removed is None:
                    raise OptimizerContextFailure(
                        "no optional memory remains after context overflow"
                    ) from exc
                omitted.append(removed)
                limit = min(math.floor(limit * 0.8), event["estimated_input_tokens"] - 1)
                stage["input_budget"] = limit
                self.save()
            finally:
                if reservation:
                    await self.budget.release(reservation)
        raise OptimizerContextFailure("optimizer context retry budget exhausted")

    async def _call(self, context, skill, selection):
        if selection:
            method = getattr(self.backend, "select_optimizer_target", None)
            if method is None:
                raise OptimizerContextFailure("backend does not support target selection")
            return await method(context, skill)
        return await self.backend.optimize_package(context, skill)

    async def optimize_package(self, context, skill):
        try:
            return await self._optimize_package(context, skill)
        except OptimizerContextFailure as exc:
            self.state["optimizer_context_error"] = str(exc)
            self.save()
            raise

    async def _optimize_package(self, context, skill):
        window, output, limit = await self.capabilities()
        source_digest = digest(context)
        request = self.state.get("optimizer_request")
        if request is None:
            payload = copy.deepcopy(context)
            payload["traces"] = _unique(payload.get("traces", []))
            payload["training_memory"] = memory_view(
                self.state.get("optimizer_memories", []), payload, self.forbidden
            )
            request = {
                "id": f"b{self.state['completed_batches'] + 1}-a{self.state['attempt']}",
                "source_digest": source_digest,
                "window": window,
                "payload": payload,
            }
            self.state["optimizer_request"] = request
            self.save()
        elif request["source_digest"] != source_digest:
            raise OptimizerContextFailure("saved optimizer request no longer matches skills/traces")
        payload = copy.deepcopy(request["payload"])
        system = self.system(payload, skill)
        while self.tokens(system, payload) > limit:
            removed = reduce_context(payload)
            if removed is None:
                break
            request.setdefault("omitted", []).append(removed)
        if "selection" in request or self.tokens(system, payload) > limit:
            editable = [
                p
                for p in context["skills"]
                if p["manifest"]["kind"] in ("router", "judge", "specialist", "workflow")
                and (not context.get("joint_optimization") or p.get("status") != "frozen")
                and p["manifest"]["name"]
                not in (
                    "generation_agent",
                    "skill_optimizer",
                    "verifier",
                    "generation_verifier",
                    "package_validator",
                    "dag_policy",
                    "budget_policy",
                    "pricing_policy",
                    "data_firewall",
                )
            ]
            catalog = [{"skill_id": p["skill_id"], "manifest": p["manifest"]} for p in editable]
            request["strategy"] = "two_stage"
            selection_context = dict(copy.deepcopy(request["payload"]), editable_catalog=catalog)
            selection_context.pop("skills", None)
            choice = await self.invoke(
                request,
                "selection",
                selection_context,
                skill,
                selection=True,
                limit=limit,
                output=output,
            )
            if context.get("joint_optimization"):
                payload = self._joint_selection_payload(choice, editable, request["payload"])
            else:
                payload = self._single_selection_payload(choice, editable, request["payload"])
            if payload is None:
                return BackendResult(
                    {"action": "no_change", "rationale": choice.get("rationale", "")}
                )
        response = await self.invoke(
            request, "proposal", payload, skill, selection=False, limit=limit, output=output
        )
        if not isinstance(response, dict):
            raise OptimizerProtocolFailure("optimizer proposal must be a JSON object")
        if "selected_target_skill_ids" in payload:
            from .joint_optimizer import response_changes

            try:
                changes = response_changes(response)
            except ValueError as exc:
                raise OptimizerProtocolFailure(str(exc)) from exc
            for change in changes:
                if (
                    change.get("action") == "edit"
                    and change.get("target_skill_id") not in payload["selected_target_skill_ids"]
                ):
                    raise OptimizerProtocolFailure("proposal changed the selected targets")
                if change.get("action") == "add" and not payload["selected_allow_additions"]:
                    raise OptimizerProtocolFailure("proposal added an unselected package")
        if (
            payload.get("selected_action") == "edit"
            and response.get("action") != "no_change"
            and (
                response.get("action") != "edit"
                or response.get("target_skill_id") != payload["selected_target_skill_id"]
            )
        ):
            raise OptimizerProtocolFailure("proposal changed the selected target")
        if payload.get("selected_action") == "add" and response.get("action") not in (
            "add",
            "no_change",
        ):
            raise OptimizerProtocolFailure("proposal changed the selected addition action")
        return BackendResult(response)

    @staticmethod
    def _joint_selection_payload(choice, editable, original):
        if not isinstance(choice, dict):
            raise OptimizerProtocolFailure("invalid joint target selection")
        if choice.get("action") != "joint":
            legacy = ManagedOptimizerBackend._single_selection_payload(choice, editable, original)
            if legacy is None:
                return None
            target = legacy.pop("selected_target_skill_id")
            action = legacy.pop("selected_action")
            legacy["selected_target_skill_ids"] = [target] if action == "edit" else []
            legacy["selected_allow_additions"] = action == "add"
            return legacy
        if set(choice) - {"action", "target_skill_ids", "allow_additions", "rationale"}:
            raise OptimizerProtocolFailure("unknown joint selection fields")
        targets = choice.get("target_skill_ids")
        additions = choice.get("allow_additions")
        if (
            not isinstance(targets, list)
            or any(not isinstance(t, str) for t in targets)
            or len(set(targets)) != len(targets)
            or not set(targets) <= {p["skill_id"] for p in editable}
            or type(additions) is not bool
            or not (targets or additions)
        ):
            raise OptimizerProtocolFailure("invalid joint target selection")
        payload = copy.deepcopy(original)
        payload["skills"] = [p for p in editable if p["skill_id"] in targets]
        payload["selected_target_skill_ids"] = targets
        payload["selected_allow_additions"] = additions
        return payload

    @staticmethod
    def _single_selection_payload(choice, editable, original):
        if not isinstance(choice, dict) or choice.get("action") not in (
            "edit",
            "add",
            "no_change",
        ):
            raise OptimizerProtocolFailure("invalid optimizer target selection")
        action, target = choice["action"], choice.get("target_skill_id")
        if action == "no_change":
            if target is not None:
                raise OptimizerProtocolFailure("no_change selection must have null target")
            return None
        if action == "edit" and (
            not isinstance(target, str) or target not in {p["skill_id"] for p in editable}
        ):
            raise OptimizerProtocolFailure("selection targeted a frozen or unknown package")
        if action == "add" and target is not None:
            raise OptimizerProtocolFailure("addition selection must have null target")
        payload = copy.deepcopy(original)
        payload["skills"] = (
            [p for p in editable if p["skill_id"] == target] if action == "edit" else []
        )
        payload["selected_action"] = action
        payload["selected_target_skill_id"] = target
        return payload
