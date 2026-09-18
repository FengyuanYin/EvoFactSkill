from __future__ import annotations


def normalize_newlines(text: str) -> str:
    """Return platform-independent text without changing any non-newline content."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_frontmatter(
    text: str,
    *,
    required: bool = True,
) -> tuple[dict[str, str], str]:
    """Parse the scalar YAML frontmatter used by Skill packages."""
    normalized = normalize_newlines(text)
    if not normalized.startswith("---\n"):
        if required:
            raise ValueError("SKILL.md must start with YAML frontmatter")
        return {}, normalized
    head, marker, body = normalized[4:].partition("\n---\n")
    if not marker:
        raise ValueError("unterminated SKILL.md frontmatter")
    metadata: dict[str, str] = {}
    for line in head.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body.strip()


def render_frontmatter(metadata: dict[str, str], body: str) -> str:
    """Render canonical LF-only Skill frontmatter."""
    header = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    return f"---\n{header}\n---\n{body.strip()}\n"
