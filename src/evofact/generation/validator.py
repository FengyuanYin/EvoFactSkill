from __future__ import annotations

from dataclasses import dataclass

from .verifier import ChallengeVerifier


@dataclass(frozen=True)
class GeneratedValidationResult:
    accepted: tuple
    rejected: tuple[dict, ...]


class GeneratedSampleValidator:
    def __init__(self):
        self._verifier = ChallengeVerifier()

    def validate(self, response, request, *, config, forbidden=(), existing=()):
        accepted, rejected = self._verifier.validate(
            response,
            request,
            config=config,
            forbidden=forbidden,
            existing=existing,
        )
        return GeneratedValidationResult(tuple(accepted), tuple(rejected))
