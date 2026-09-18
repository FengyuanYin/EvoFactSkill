async def run_ablations(runner, samples=None):
    """Reject the retired alias-based ablation implementation.

    Named ablation arms must have independent execution switches and are intentionally
    unavailable until those switches exist. Routing strategies are controls, not ablations.
    """
    del runner, samples
    raise RuntimeError(
        "ablation is disabled until every named arm has an independent implementation"
    )
