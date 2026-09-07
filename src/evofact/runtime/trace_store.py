import json
from dataclasses import asdict
from pathlib import Path
from evofact.core.models import InferenceTrace

class TraceStore:
    def __init__(self,path:Path): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,trace:InferenceTrace)->None:
        with self.path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(asdict(trace),ensure_ascii=False,default=str)+"\n")
