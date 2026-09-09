from evofact.routing.strategies import STRATEGIES


async def run_ablations(runner, samples=None):
    results = {}
    for arm in STRATEGIES:
        strategy = {
            "single-llm": "static",
            "prompt-only-evolution": "static",
            "no-discovery": "utility-aware",
            "no-negative-transfer": "utility-aware",
            "full": "utility-aware",
        }.get(arm, arm)
        traces, evaluation = await runner.run(samples=samples, strategy=strategy)
        results[arm] = {"n_traces": len(traces), "metrics": evaluation.aggregate_metrics}
    return results
