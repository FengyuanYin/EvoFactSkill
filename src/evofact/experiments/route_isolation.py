"""Keep each evolution command's persistent artifacts in a distinct namespace."""

from dataclasses import replace
from pathlib import Path


def route_artifact_path(path, mode):
    path = Path(path)
    return path if path.parent.name == mode else path.parent / mode / path.name


def evolution_route_config(config, mode):
    if mode == "evolve":
        return config
    if mode not in {"meta-evolve", "adversarial-evolve"}:
        raise ValueError("unknown evolution route")

    def directory(path):
        path = Path(path)
        return path if path.name == mode else path / mode

    return replace(
        config,
        skill_store=directory(config.skill_store),
        output_dir=directory(config.output_dir),
        meta_learning=replace(
            config.meta_learning,
            checkpoint_path=route_artifact_path(config.meta_learning.checkpoint_path, mode),
        ),
        execution=replace(
            config.execution,
            trace_log=route_artifact_path(config.execution.trace_log, mode)
            if config.execution.trace_log
            else None,
        ),
        generation=replace(
            config.generation, store_path=route_artifact_path(config.generation.store_path, mode)
        ),
    )
