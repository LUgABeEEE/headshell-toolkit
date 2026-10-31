"""TripoSR reconstruct adapter: one image -> OBJ mesh (MIT).

The 3D model itself runs elsewhere (e.g. the RTX 5060 box); this adapter only
builds the command, runs it, and validates the produced mesh.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Callable

from pipeline.contracts import Artifact

CommandRunner = Callable[[list[str], int], int]


@dataclass(frozen=True)
class TripoSRConfig:
    python: str = "python"
    script: str = "run.py"
    workdir: str = ""
    timeout: int = 1800
    extra_args: tuple[str, ...] = ()


class TripoSRAdapter:
    """StageAdapter for the `reconstruct` stage.

    context must provide workdir; the source image comes from context["input_image"]
    or, when absent, from the manifest's `design` artifact.
    """

    stage = "reconstruct"

    def __init__(self, config: TripoSRConfig, runner: CommandRunner | None = None):
        self.config = config
        self._runner = runner

    def build_command(self, image: str | Path, out_obj: str | Path) -> list[str]:
        """Return the argv for one reconstruction run.

        `out_obj` is the output directory (directory-like path) passed as --output-dir.
        """
        argv = [
            self.config.python,
            self.config.script,
            "--image",
            str(image),
            "--output-dir",
            str(Path(out_obj)),
        ]
        return argv + list(self.config.extra_args)

    def _real_runner(self, cwd: Path) -> CommandRunner:
        """Build the default subprocess runner bound to a working directory."""

        def _run(cmd: list[str], timeout: int) -> int:
            completed = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, check=False)
            return int(completed.returncode)

        return _run

    def run(self, context: dict) -> Artifact:
        """Run the command, then require a non-empty OBJ; return an Artifact.

        Raise RuntimeError on a non-zero exit code or a missing/empty OBJ.
        """
        workdir = Path(context["workdir"])
        image = context.get("input_image")
        if not image:
            manifest = context.get("manifest")
            design = manifest.artifacts.get("design") if manifest is not None else None
            image = design.path if design is not None else None
        if not image:
            raise RuntimeError("no input image")

        out_obj = workdir / "reconstruct" / "mesh.obj"
        out_dir = out_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = self.build_command(image, out_dir)
        runner = self._runner or self._real_runner(Path(self.config.workdir) if self.config.workdir else workdir)
        code = runner(cmd, self.config.timeout)
        if code != 0:
            raise RuntimeError(f"triposr exited with code {code}")
        if not out_obj.exists() or out_obj.stat().st_size == 0:
            raise RuntimeError(f"missing or empty OBJ: {out_obj}")
        return Artifact.create("reconstruct", out_obj, {"adapter": "triposr", "exit_code": code})
