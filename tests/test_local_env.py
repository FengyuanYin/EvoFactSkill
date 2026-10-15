from evofact.local_env import load_local_env


def test_local_defaults_preserve_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOFACT_TEST_DEFAULT", raising=False)
    monkeypatch.setenv("EVOFACT_TEST_OVERRIDE", "explicit")
    path = tmp_path / ".env"
    path.write_text(
        '# local defaults\nEVOFACT_TEST_DEFAULT="local"\nEVOFACT_TEST_OVERRIDE=file\n',
        encoding="utf-8-sig",
    )
    load_local_env(path)
    import os

    assert os.environ["EVOFACT_TEST_DEFAULT"] == "local"
    assert os.environ["EVOFACT_TEST_OVERRIDE"] == "explicit"


def test_missing_local_env_is_optional(tmp_path):
    load_local_env(tmp_path / "missing.env")


def test_invalid_entry_does_not_expose_contents(tmp_path):
    import pytest

    path = tmp_path / ".env"
    path.write_text("private-value-without-assignment", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1") as error:
        load_local_env(path)
    assert "private-value" not in str(error.value)
