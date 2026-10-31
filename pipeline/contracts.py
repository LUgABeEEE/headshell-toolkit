"""Pipeline contracts (MIT).

Stages are explicit and ordered. Every stage emits an Artifact with a content hash so the
runner can resume without recomputing completed work.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib, json

STAGES = ("design", "reconstruct", "shell", "verify", "package")
STATUSES = ("pending", "running", "failed", "completed")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Artifact:
    stage: str
    path: str
    sha256: str
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage: {self.stage}")
        if not self.path:
            raise ValueError("artifact path is empty")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be 64 hex chars")

    @classmethod
    def create(cls, stage: str, path: str | Path, meta: dict | None = None) -> "Artifact":
        a = cls(stage=stage, path=str(path), sha256=file_sha256(path), meta=dict(meta or {}))
        a.validate()
        return a


@dataclass
class Manifest:
    job_id: str
    version: int = 2
    status: str = "pending"
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    error: str | None = None

    def add(self, artifact: Artifact) -> None:
        artifact.validate()
        self.artifacts[artifact.stage] = artifact

    def to_json(self) -> str:
        data = {
            "job_id": self.job_id,
            "version": self.version,
            "status": self.status,
            "error": self.error,
            "artifacts": {k: asdict(v) for k, v in self.artifacts.items()},
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        d = json.loads(text)
        if d.get("version") != 2:
            raise ValueError("unsupported manifest version")
        if d.get("status") not in STATUSES:
            raise ValueError("invalid status")
        m = cls(job_id=d["job_id"], version=2, status=d["status"], error=d.get("error"))
        for k, v in (d.get("artifacts") or {}).items():
            a = Artifact(stage=v["stage"], path=v["path"], sha256=v["sha256"], meta=v.get("meta") or {})
            a.validate()
            if a.stage != k:
                raise ValueError("artifact key/stage mismatch")
            m.artifacts[k] = a
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def is_done(self, stage: str) -> bool:
        a = self.artifacts.get(stage)
        return bool(a and a.sha256 and Path(a.path).exists() and file_sha256(a.path) == a.sha256)
