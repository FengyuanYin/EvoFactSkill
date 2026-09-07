import hashlib, json, re
from pathlib import Path, PurePosixPath
from evofact.core.models import SkillKind, SkillScope, SkillSpec, SkillStatus, Trigger

def _frontmatter(text: str) -> tuple[dict[str,str],str]:
    if not text.startswith("---\n"): raise ValueError("SKILL.md must start with YAML frontmatter")
    head, marker, body=text[4:].partition("\n---\n")
    if not marker: raise ValueError("unterminated frontmatter")
    meta={}
    for line in head.splitlines():
        key,sep,value=line.partition(":")
        if sep: meta[key.strip()]=value.strip().strip("\"'")
    return meta,body.strip()

def _safe_path(rel: str) -> None:
    path=PurePosixPath(rel.replace("\\","/"))
    if path.is_absolute() or ".." in path.parts or not path.parts: raise ValueError(f"unsafe skill resource path: {rel}")

def load_skill_package(directory: Path, *, status: SkillStatus=SkillStatus.ACTIVE) -> SkillSpec:
    meta,instructions=_frontmatter((directory/"SKILL.md").read_text(encoding="utf-8"))
    name=meta.get("name","").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}",name): raise ValueError(f"invalid skill name: {name}")
    extra={}
    if (directory/"metadata.json").is_file(): extra=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    resources={}
    for folder in ("references","templates","assets","scripts"):
        base=directory/folder
        if not base.exists(): continue
        for path in base.rglob("*"):
            if path.is_file():
                rel=path.relative_to(directory).as_posix(); _safe_path(rel)
                if path.stat().st_size>200_000: raise ValueError(f"resource too large: {rel}")
                resources[rel]=path.read_text(encoding="utf-8")
    payload=json.dumps({"meta":meta,"extra":extra,"instructions":instructions,"resources":resources},sort_keys=True,ensure_ascii=False)
    sid=hashlib.sha256(payload.encode()).hexdigest()[:20]
    scope=SkillScope(tuple(extra.get("domains",[])),tuple(extra.get("datasets",[])),tuple(extra.get("temporal_windows",[])),tuple(extra.get("tags",[])))
    triggers=tuple(Trigger(**x) for x in extra.get("triggers",[]))
    return SkillSpec(sid,name,SkillKind(meta.get("kind","specialist")),meta.get("version","0.1.0"),status,instructions,resources,scope,triggers,tuple(extra.get("parent_ids",[])),extra.get("safety_level","text_only"))
