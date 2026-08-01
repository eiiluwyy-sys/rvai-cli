"""Typer entry point for RVAI."""

from pathlib import Path

import typer

from rvai.compatibility import (
    CompatibilityError,
    CompatibilityMatcher,
    load_hardware_profile,
)
from rvai.hardware import HardwareProbe, HardwareProbeError
from rvai.planner import RunPlanner
from rvai.registry import ModelRegistry, RegistryError

app = typer.Typer(
    name="rvai",
    help="Inspect and plan RISC-V AI workloads.",
    no_args_is_help=True,
)


def _registry() -> ModelRegistry:
    return ModelRegistry()


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


@app.command("list")
def list_models() -> None:
    """List all registered model names."""

    try:
        manifests = _registry().list()
    except RegistryError as exc:
        _fail(str(exc))
    for manifest in manifests:
        typer.echo(manifest.name)


@app.command()
def show(model: str = typer.Argument(..., help="Registered model name.")) -> None:
    """Print one validated Manifest as JSON."""

    try:
        manifest = _registry().get(model)
    except RegistryError as exc:
        _fail(str(exc))
    typer.echo(manifest.model_dump_json(indent=2))


@app.command()
def run(
    model: str = typer.Argument(..., help="Registered model name."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate a plan without executing the workload.",
    ),
) -> None:
    """Generate a structured model execution plan."""

    if not dry_run:
        _fail("RVAI V0.1 only supports 'run' with --dry-run")

    try:
        manifest = _registry().get(model)
        plan = RunPlanner().plan(manifest, dry_run=True)
    except (RegistryError, ValueError) as exc:
        _fail(str(exc))
    typer.echo(plan.model_dump_json(indent=2))


@app.command()
def detect() -> None:
    """Detect the current hardware and runtime environment."""

    try:
        profile = HardwareProbe().detect()
    except HardwareProbeError as exc:
        _fail(str(exc))
    typer.echo(profile.model_dump_json(indent=2))


@app.command("check")
def check_model(
    model: str = typer.Argument(..., help="Registered model name."),
    profile_path: Path | None = typer.Option(
        None,
        "--profile",
        help="Read a Hardware Profile from JSON instead of probing this host.",
    ),
) -> None:
    """Check model compatibility and immediate execution readiness."""

    try:
        manifest = _registry().get(model)
        hardware = (
            load_hardware_profile(profile_path)
            if profile_path is not None
            else HardwareProbe().detect()
        )
        report = CompatibilityMatcher().check(manifest, hardware)
    except (CompatibilityError, HardwareProbeError, RegistryError) as exc:
        _fail(str(exc))
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
