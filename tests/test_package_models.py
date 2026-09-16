from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import pytest

from evofact.core.package_models import SkillContract, SkillFile


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_skill_file_accepts_valid_content() -> None:
    content = b"# Evidence Specialist\n"

    skill_file = SkillFile(
        path="SKILL.md",
        media_type="text/markdown",
        content=content,
        digest=_digest(content),
    )

    assert skill_file.path == "SKILL.md"
    assert skill_file.content == content
    assert skill_file.schema_version == "skill_file_v1"


def test_skill_file_accepts_nested_portable_path() -> None:
    content = b"print('ok')\n"

    skill_file = SkillFile(
        path="scripts/check.py",
        media_type="text/x-python",
        content=content,
        digest=_digest(content),
        executable=True,
    )

    assert skill_file.executable is True


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/SKILL.md",
        "C:/skills/SKILL.md",
        "../SKILL.md",
        "references/../SKILL.md",
        "./SKILL.md",
        "scripts\\check.py",
        "references//guide.md",
        "references/",
        " SKILL.md",
        "SKILL.md ",
    ],
)
def test_skill_file_rejects_unsafe_paths(path: str) -> None:
    content = b"content"

    with pytest.raises(ValueError):
        SkillFile(
            path=path,
            media_type="text/plain",
            content=content,
            digest=_digest(content),
        )


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "abc",
        "0" * 63,
        "0" * 65,
        "G" * 64,
        "A" * 64,
    ],
)
def test_skill_file_rejects_invalid_digest_format(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        SkillFile(
            path="SKILL.md",
            media_type="text/markdown",
            content=b"content",
            digest=digest,
        )


def test_skill_file_rejects_digest_content_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        SkillFile(
            path="SKILL.md",
            media_type="text/markdown",
            content=b"actual content",
            digest=_digest(b"different content"),
        )


def test_skill_file_rejects_empty_media_type() -> None:
    content = b"content"

    with pytest.raises(ValueError, match="media_type"):
        SkillFile(
            path="SKILL.md",
            media_type=" ",
            content=content,
            digest=_digest(content),
        )


def test_skill_file_rejects_non_bytes_content() -> None:
    content = "not bytes"

    with pytest.raises(ValueError, match="content must be bytes"):
        SkillFile(
            path="SKILL.md",
            media_type="text/markdown",
            content=content,  # type: ignore[arg-type]
            digest=_digest(content.encode()),
        )


def test_skill_file_is_immutable() -> None:
    content = b"content"
    skill_file = SkillFile(
        path="SKILL.md",
        media_type="text/markdown",
        content=content,
        digest=_digest(content),
    )

    with pytest.raises(FrozenInstanceError):
        skill_file.path = "changed.md"  # type: ignore[misc]


def test_skill_contract_accepts_declared_interfaces() -> None:
    contract = SkillContract(
        consumes=("sample_public",),
        produces=("specialist_report",),
        requires_capabilities=("llm",),
        optional_capabilities=("web_search",),
        allow_root=False,
        allow_parallel=True,
        output_schema="specialist_report_v1",
    )

    assert contract.consumes == ("sample_public",)
    assert contract.produces == ("specialist_report",)
    assert contract.allow_root is False


def test_skill_contract_allows_empty_optional_declarations() -> None:
    assert SkillContract() == SkillContract()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("consumes", ("",)),
        ("produces", (" ",)),
        ("requires_capabilities", ("llm", "")),
        ("optional_capabilities", ("\t",)),
    ],
)
def test_skill_contract_rejects_empty_items(
    field_name: str,
    value: tuple[str, ...],
) -> None:
    arguments = {field_name: value}

    with pytest.raises(ValueError, match=field_name):
        SkillContract(**arguments)


def test_skill_contract_rejects_blank_output_schema() -> None:
    with pytest.raises(ValueError, match="output_schema"):
        SkillContract(output_schema=" ")
