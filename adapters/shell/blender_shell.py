"""Blender shell adapter: reconstructed mesh -> printable shell (MIT).

Wraps the M1b kernel (core/print/build_shell_full.py): writes a config JSON,
runs Blender headless, then hard-gates the result with pipeline.verify.verify_stl.
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.contracts import Artifact
from pipeline.verify import verify_stl

CommandRunner = Callable[[list[str], int], int]


@dataclass(frozen=True)
class BlenderShellConfig:
    blender: str = "blender"
    script: str = ""
    params: dict = None  # type: ignore[assignment]
    timeout: int = 1800
    require_watertight: bool = True

    def __post_init__(self):
        if self.params is None:
            object.__setattr__(self, "params", {})


class BlenderShellAdapter:
    """StageAdapter for the `shell` stage.

    context must provide workdir; the outer mesh comes from context["outer_mesh"]
    or the manifest's `reconstruct` artifact.
    """

    stage = "shell"

    def __init__(self, config: BlenderShellConfig, runner: CommandRunner | None = None):
        self.config = config
        self._runner = runner

    def build_config(self, outer: str | Path, out_stl: str | Path, inner: str | None) -> dict:
        """Merge config.params with outer/inner/out paths into the kernel config dict."""
        cfg = dict(self.config.params)
        cfg["outer"] = str(outer)
        if inner is not None:
            cfg["inner"] = str(inner)
        else:
            cfg.pop("inner", None)
        cfg["out"] = str(out_stl)
        return cfg

    def build_command(self, config_path: str | Path) -> list[str]:
        """Return the headless Blender argv for one run."""
        return [
            self.config.blender,
            "-b",
            "--factory-startup",
            "-P",
            self.config.script,
            "--",
            "--config",
            str(config_path),
        ]

    def _real_runner(self, cwd: Path) -> CommandRunner:
        """Build the default subprocess runner bound to a working directory."""

        def _run(cmd: list[str], timeout: int) -> int:
            completed = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, check=False)
            return int(completed.returncode)

        return _run

    def run(self, context: dict) -> Artifact:
        """Write the config, run Blender, then verify the STL; return an Artifact.

        Raise RuntimeError on a non-zero exit, a missing STL, or a failed verify.
        """
        workdir = Path(context["workdir"])
        outer = context.get("outer_mesh")
        if not outer:
            manifest = context.get("manifest")
            reconstruct = manifest.artifacts.get("reconstruct") if manifest is not None else None
            outer = reconstruct.path if reconstruct is not None else None
        if not outer:
            raise RuntimeError("no outer mesh")
        inner = context.get("inner_mesh")

        out_stl = workdir / "shell.stl"
        config_path = workdir / "shell_config.json"
        config_path.write_text(
            json.dumps(self.build_config(outer, out_stl, inner), ensure_ascii=False),
            encoding="utf-8",
        )

        runner = self._runner or self._real_runner(workdir)
        code = runner(self.build_command(config_path), self.config.timeout)
        if code != 0:
            raise RuntimeError(f"blender shell exited with code {code}")

        report = verify_stl(
            out_stl,
            expected_bbox_mm=self.config.params.get("expected_bbox_mm"),
            bbox_tolerance_mm=self.config.params.get("bbox_tolerance_mm", 1.0),
            require_watertight=self.config.require_watertight,
        )
        if not out_stl.is_file() or not report.ok:
            detail = "; ".join(report.errors) or f"missing output STL: {out_stl}"
            raise RuntimeError(f"shell STL verification failed: {detail}")
        return Artifact.create("shell", out_stl, {
            "adapter": "blender_shell",
            "watertight": report.checks.get("watertight"),
            "verify_errors": report.errors,
        })
