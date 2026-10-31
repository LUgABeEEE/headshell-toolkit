"""Stage runner with resume and content-hash skip (MIT)."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
from pipeline.contracts import Artifact, Manifest, STAGES

class StageAdapter(Protocol):
    def run(self, context: dict) -> Artifact: ...

class PipelineRunner:
    def __init__(self, manifest_path: str | Path, adapters: dict[str, StageAdapter]):
        self.manifest_path = Path(manifest_path)
        self.adapters = dict(adapters)

    def run(self, order: tuple[str, ...] = STAGES) -> Manifest:
        """Run stages in order, skipping stages already complete with a matching hash.
        On failure: manifest.status='failed', error set, save, re-raise."""
        missing = [stage for stage in order if stage not in self.adapters]
        if missing:
            raise ValueError(f"no adapter registered for stage(s): {missing}")
        manifest = (
            Manifest.load(self.manifest_path)
            if self.manifest_path.exists()
            else Manifest(job_id=self.manifest_path.stem)
        )
        manifest.status = "running"
        manifest.error = None
        manifest.save(self.manifest_path)
        for stage in order:
            if manifest.is_done(stage):
                continue
            try:
                artifact = self.adapters[stage].run(
                    {
                        "stage": stage,
                        "manifest": manifest,
                        "workdir": str(self.manifest_path.parent),
                    }
                )
                if not isinstance(artifact, Artifact):
                    raise TypeError(
                        f"adapter for stage {stage!r} returned "
                        f"{type(artifact).__name__}, expected Artifact"
                    )
                manifest.add(artifact)
                manifest.save(self.manifest_path)
            except Exception as exc:
                manifest.status = "failed"
                manifest.error = str(exc)
                manifest.save(self.manifest_path)
                raise
        manifest.status = "completed"
        manifest.save(self.manifest_path)
        return manifest
