from __future__ import annotations

import hashlib, json, os, tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from evofact.core.models import SkillKind, SkillScope, SkillSpec, SkillStatus, Trigger

def _jsonable(value):
    if isinstance(value, dict): return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_jsonable(v) for v in value]
    if hasattr(value,"value"): return value.value
    return value

def _skill_from(data: dict) -> SkillSpec:
    data=dict(data); data["kind"]=SkillKind(data["kind"]); data["status"]=SkillStatus(data["status"])
    data["scope"]=SkillScope(**{k:tuple(v) for k,v in data.get("scope",{}).items()}); data["triggers"]=tuple(Trigger(**x) for x in data.get("triggers",[])); data["parent_ids"]=tuple(data.get("parent_ids",[]))
    return SkillSpec(**data)

class SkillRepository:
    def __init__(self, root: Path):
        self.root=Path(root); self.snapshots=self.root/"snapshots"; self.events=self.root/"events.jsonl"; self.active_file=self.root/"active.json"
        self.snapshots.mkdir(parents=True,exist_ok=True); self.root.mkdir(parents=True,exist_ok=True)
    def _atomic_json(self,path:Path,value) -> None:
        fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".tmp-",suffix=".json")
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(_jsonable(value),stream,ensure_ascii=False,sort_keys=True,indent=2)
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def _event(self,action:str,**payload) -> None:
        event={"schema_version":"skill_event_v1","timestamp":datetime.now(timezone.utc).isoformat(),"action":action,**_jsonable(payload)}
        with self.events.open("a",encoding="utf-8") as stream: stream.write(json.dumps(event,ensure_ascii=False,sort_keys=True)+"\n")
    def _active(self)->dict[str,str]:
        return json.loads(self.active_file.read_text(encoding="utf-8")) if self.active_file.exists() else {}
    def save(self,skill:SkillSpec,action:str="stage") -> str:
        data=_jsonable(asdict(skill)); digest=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest(); path=self.snapshots/f"{digest}.json"
        if not path.exists(): self._atomic_json(path,data)
        self._event(action,skill_id=skill.skill_id,snapshot=digest,name=skill.name,status=skill.status)
        return digest
    def get_snapshot(self,digest:str)->SkillSpec:
        return _skill_from(json.loads((self.snapshots/f"{digest}.json").read_text(encoding="utf-8")))
    def list(self,status:SkillStatus|None=None)->list[SkillSpec]:
        result=[self.get_snapshot(p.stem) for p in sorted(self.snapshots.glob("*.json"))]
        return [s for s in result if status is None or s.status==status]
    def active(self)->dict[str,SkillSpec]: return {n:self.get_snapshot(d) for n,d in self._active().items()}
    def promote(self,skill:SkillSpec)->str:
        if skill.status==SkillStatus.RETIRED: raise ValueError("cannot promote retired skill")
        promoted=replace(skill,status=SkillStatus.ACTIVE); digest=self.save(promoted,"promote"); active=self._active(); active[promoted.name]=digest; self._atomic_json(self.active_file,active); return digest
    def promote_batch(self, skills:list[SkillSpec]) -> list[str]:
        promoted=[]; digests=[]
        for skill in skills:
            if skill.status==SkillStatus.RETIRED: raise ValueError("cannot promote retired skill")
            item=replace(skill,status=SkillStatus.ACTIVE); data=_jsonable(asdict(item)); digest=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            promoted.append(item); digests.append(digest)
        for item,digest in zip(promoted,digests):
            path=self.snapshots/f"{digest}.json"
            if not path.exists(): self._atomic_json(path,_jsonable(asdict(item)))
        active=self._active()
        for item,digest in zip(promoted,digests): active[item.name]=digest
        self._atomic_json(self.active_file,active)
        for item,digest in zip(promoted,digests): self._event("meta_promote",skill_id=item.skill_id,snapshot=digest,name=item.name,status=item.status)
        return digests
    def freeze(self,name:str)->str:
        skill=self.active()[name]; frozen=replace(skill,status=SkillStatus.FROZEN); digest=self.save(frozen,"freeze"); active=self._active(); active[name]=digest; self._atomic_json(self.active_file,active); return digest
    def retire(self,name:str,reason:str)->str:
        skill=self.active()[name]; retired=replace(skill,status=SkillStatus.RETIRED); digest=self.save(retired,"retire"); active=self._active(); active.pop(name,None); self._atomic_json(self.active_file,active); self._event("retire_reason",name=name,reason=reason); return digest
    def rollback(self,name:str,digest:str)->None:
        skill=self.get_snapshot(digest)
        if skill.name!=name or skill.status==SkillStatus.RETIRED: raise ValueError("invalid rollback target")
        active=self._active(); active[name]=digest; self._atomic_json(self.active_file,active); self._event("rollback",name=name,snapshot=digest)
    def history(self)->list[dict]:
        return [json.loads(x) for x in self.events.read_text(encoding="utf-8").splitlines()] if self.events.exists() else []
