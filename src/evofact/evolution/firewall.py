from __future__ import annotations

import json
from collections.abc import Sequence

from evofact.core.models import AttributionReport, DomainEpisode, EvolutionProposal, InferenceTrace, MetaTrainView, Sample


class CandidateFirewall:
    def __init__(self, samples: Sequence[Sample], final_test_domains: Sequence[str] = ()):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.final_test_domains = set(final_test_domains)

    def build_generation_view(self, episode: DomainEpisode, traces: Sequence[InferenceTrace], attributions: Sequence[AttributionReport]) -> MetaTrainView:
        if set(episode.final_test_domains) != self.final_test_domains:
            raise ValueError("firewall final-test domain policy mismatch")
        allowed_ids = set(episode.meta_train_sample_ids)
        selected = tuple(trace for trace in traces if str(trace.sample_public.get("sample_id")) in allowed_ids)
        if len(selected) != len(traces):
            raise ValueError("generation traces contain non-meta-train samples")
        allowed_trace_ids = {trace.trace_id for trace in selected}
        if any(report.trace_id not in allowed_trace_ids for report in attributions):
            raise ValueError("attribution contains non-meta-train trace")
        summaries = tuple({"trace_id": report.trace_id, "error_types": tuple(error.value for error in report.error_types), "responsible_skill_ids": report.responsible_skill_ids, "confidence": report.confidence} for report in attributions)
        return MetaTrainView(episode.episode_id, tuple(trace.trace_id for trace in selected), tuple(trace.sample_public for trace in selected), summaries)

    def validate_candidate(self, candidate: EvolutionProposal, episode: DomainEpisode) -> None:
        if set(episode.final_test_domains) != self.final_test_domains:
            raise ValueError("firewall final-test domain policy mismatch")
        forbidden = [self.samples[sid] for sid in episode.meta_test_sample_ids if sid in self.samples]
        forbidden += [sample for sample in self.samples.values() if (sample.domain or sample.dataset) in self.final_test_domains]
        blob = json.dumps(candidate.model_dump(), ensure_ascii=False).casefold()
        for sample in forbidden:
            tokens = [sample.sample_id.casefold()]
            normalized_text = " ".join(sample.text.casefold().split())
            if len(normalized_text) >= 12:
                tokens.append(normalized_text)
            if any(token and token in blob for token in tokens):
                raise ValueError(f"candidate leaks held-out sample {sample.sample_id}")
