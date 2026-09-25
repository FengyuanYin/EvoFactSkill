from dataclasses import replace
from pathlib import Path

from evofact.config import OptimizerBackendConfig, load_config
from evofact.experiments.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_backend_uses_independent_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("FORWARD_KEY", "forward-secret")
    monkeypatch.setenv("OPTIMIZER_KEY", "optimizer-secret")
    monkeypatch.setenv("OPTIMIZER_MODEL", "advanced-model")
    monkeypatch.setenv("OPTIMIZER_URL", "https://optimizer.example/v1")
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(
        config,
        backend="openai-compatible",
        model="cheap-model",
        base_url="https://forward.example/v1",
        api_key_env="FORWARD_KEY",
        pricing=replace(config.pricing, require_cost_for_promotion=False),
        optimizer_backend=OptimizerBackendConfig(
            enabled=True,
            model_env="OPTIMIZER_MODEL",
            base_url_env="OPTIMIZER_URL",
            api_key_env="OPTIMIZER_KEY",
            temperature=1.0,
        ),
    )
    runner = ExperimentRunner(config, ROOT)

    forward = runner._backend()
    optimizer = runner._optimizer_backend()

    assert forward.model == "cheap-model"
    assert forward.url == "https://forward.example/v1/chat/completions"
    assert forward.temperature == 0
    assert optimizer.model == "advanced-model"
    assert optimizer.url == "https://optimizer.example/v1/chat/completions"
    assert optimizer.temperature == 1.0


def test_optimizer_backend_allows_forward_endpoint_reuse(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_KEY", "optimizer-secret")
    config = load_config(ROOT / "configs/dry_run.yaml")
    config = replace(
        config,
        base_url="https://same.example/v1",
        pricing=replace(config.pricing, require_cost_for_promotion=False),
        optimizer_backend=OptimizerBackendConfig(
            enabled=True,
            model="advanced-model",
            base_url="https://same.example/v1/",
            api_key_env="OPTIMIZER_KEY",
            require_distinct_base_url=True,
        ),
    )

    optimizer = ExperimentRunner(config, ROOT)._optimizer_backend()

    assert optimizer.model == "advanced-model"
    assert optimizer.url == "https://same.example/v1/chat/completions"


def test_weibo_config_declares_environment_backed_optimizer():
    config = load_config(ROOT / "configs/weibo21_cross_domain.yaml")

    assert config.model == "deepseek-flash"
    assert config.optimizer_backend.enabled
    assert config.optimizer_backend.model_env == "EVOFACT_OPTIMIZER_MODEL"
    assert config.optimizer_backend.base_url_env == "EVOFACT_OPTIMIZER_BASE_URL"
    assert config.optimizer_backend.api_key_env == "EVOFACT_OPTIMIZER_API_KEY"
    assert config.optimizer_backend.pricing_table_path_env == ("EVOFACT_OPTIMIZER_PRICING_TABLE")


def test_weibo_optimizer_accepts_same_endpoint_and_deepseek_pricing(monkeypatch):
    monkeypatch.setenv("EVOFACT_OPTIMIZER_MODEL", "deepseek-flash")
    monkeypatch.setenv("EVOFACT_OPTIMIZER_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("EVOFACT_OPTIMIZER_API_KEY", "test-key")
    monkeypatch.setenv(
        "EVOFACT_OPTIMIZER_PRICING_TABLE",
        "pricing/deepseek-2026-09-18-peak.json",
    )
    config = load_config(ROOT / "configs/weibo21_cross_domain.yaml")

    optimizer = ExperimentRunner(config, ROOT)._optimizer_backend()

    assert optimizer.url == "https://api.deepseek.com/v1/chat/completions"
    assert optimizer.provider == "deepseek"
    assert optimizer.pricing_table.provider == "deepseek"
