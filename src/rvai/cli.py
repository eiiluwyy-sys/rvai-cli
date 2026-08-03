"""Typer entry point for RVAI."""

import json
from pathlib import Path

import typer

from rvai.adapters import AdapterError, BenchmarkResult, BuiltinAdapter
from rvai.adapters.builtin import BuiltinAdapter as BuiltinAdapterDefaults
from rvai.artifacts import (
    ArtifactCache,
    ArtifactCacheError,
    ArtifactDownloadError,
    ArtifactDownloader,
    ArtifactIntegrityError,
    ArtifactResolver,
    PullResult,
)
from rvai.compatibility import (
    CompatibilityError,
    CompatibilityMatcher,
    load_hardware_profile,
)
from rvai.hardware import HardwareProbe, HardwareProbeError, HardwareProfile
from rvai.planner import RunPlanner
from rvai.registry import ModelRegistry, RegistryError
from rvai.results import (
    ReportFormat,
    ReportRenderError,
    ResultStoreError,
    compare_run_records,
    create_run_record,
    digest_manifest,
    load_run_record,
    render_markdown,
    save_markdown_report,
    save_run_record,
)
from rvai.targets import TargetName, create_target

app = typer.Typer(
    name="rvai",
    help="Inspect and plan RISC-V AI workloads.",
    no_args_is_help=True,
)


def _registry() -> ModelRegistry:
    return ModelRegistry()


def _hardware_probe() -> HardwareProbe:
    return HardwareProbe()


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
    """Print one validated Manifest and its local artifact cache status."""

    try:
        manifest = _registry().get(model)
        status = ArtifactResolver().status(manifest)
    except (ArtifactCacheError, RegistryError) as exc:
        _fail(str(exc))
    payload = manifest.model_dump(mode="json")
    declared_artifact = payload.get("artifact") or {}
    payload["artifact"] = {
        **declared_artifact,
        **status.model_dump(mode="json", exclude_none=True),
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def pull(
    model: str = typer.Argument(..., help="Registered model name."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Safely replace an existing cached artifact.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help="Override the model artifact cache root.",
    ),
) -> None:
    """Download and verify one Manifest-declared model artifact."""

    try:
        manifest = _registry().get(model)
    except RegistryError as exc:
        _fail(str(exc))
    spec = manifest.artifact
    if spec is None:
        _fail(f"Model '{model}' does not declare a downloadable artifact")

    cache = (
        ArtifactCache(root=cache_dir)
        if cache_dir is not None
        else ArtifactCache()
    )
    destination = cache.artifact_path(model, spec)
    try:
        downloaded = ArtifactDownloader(cache=cache).download(
            model,
            spec,
            destination,
            manifest_digest=digest_manifest(manifest),
            force=force,
        )
    except ArtifactIntegrityError as exc:
        detail = str(exc)
        if detail.startswith("Cached artifact"):
            _fail(
                f"Cached artifact for '{model}' failed verification. "
                "Use --force to replace it."
            )
        if "SHA-256" in detail:
            _fail(f"Artifact SHA-256 mismatch for '{model}'")
        _fail(f"Artifact integrity check failed for '{model}': {detail}")
    except ArtifactDownloadError as exc:
        detail = str(exc).removeprefix("Cannot download artifact: ")
        _fail(f"Failed to download artifact for '{model}': {detail}")
    except ArtifactCacheError as exc:
        _fail(f"Cannot manage artifact cache for '{model}': {exc}")

    result = PullResult(
        status=downloaded.status,
        model=downloaded.model,
        path=downloaded.path,
        sha256=downloaded.sha256,
        size_bytes=downloaded.size_bytes,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def run(
    model: str = typer.Argument(..., help="Registered model name."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate a plan without executing the workload.",
    ),
    target: TargetName = typer.Option(
        TargetName.NATIVE,
        "--target",
        help="Execution target for real runs or dry-run plans.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Save a versioned RunRecord JSON file.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing --output file.",
    ),
) -> None:
    """Plan or execute a registered workload."""

    try:
        manifest = _registry().get(model)
    except RegistryError as exc:
        _fail(str(exc))

    if force and output is None:
        _fail("--force requires --output")
    if dry_run and output is not None:
        _fail("--output is available only for completed benchmark runs")

    if dry_run:
        try:
            plan = RunPlanner().plan(
                manifest,
                dry_run=True,
                target=target.value,
            )
        except ValueError as exc:
            _fail(str(exc))
        typer.echo(plan.model_dump_json(indent=2))
        return

    if manifest.runtime != "builtin":
        _fail("Real execution is currently supported only for builtin workloads")

    try:
        execution_target = create_target(target)
        result = BuiltinAdapter().execute(manifest, target=execution_target)
    except AdapterError as exc:
        _fail(str(exc))

    if output is not None:
        try:
            hardware_profile = _detect_optional_hardware_profile()
            record = create_run_record(
                command=_reproducible_run_command(model, target, result),
                manifest=manifest,
                target=target.value,
                hardware_profile=hardware_profile,
                result=result,
            )
            save_run_record(record, output, force=force)
        except ResultStoreError as exc:
            _fail(str(exc))
    typer.echo(result.model_dump_json(indent=2))


def _detect_optional_hardware_profile() -> HardwareProfile | None:
    """Capture hardware when possible without invalidating a successful run."""

    try:
        return _hardware_probe().detect()
    except HardwareProbeError:
        return None


def _reproducible_run_command(
    model: str,
    target: TargetName,
    result: BenchmarkResult,
) -> list[str]:
    """Describe the command, including controlled non-default GEMM sizing."""

    command = ["rvai", "run", model, "--target", target.value]
    dimensions = result.matrix
    iterations = result.iterations
    defaults = (
        BuiltinAdapterDefaults.DEFAULT_M,
        BuiltinAdapterDefaults.DEFAULT_N,
        BuiltinAdapterDefaults.DEFAULT_K,
        BuiltinAdapterDefaults.DEFAULT_ITERATIONS,
    )
    actual = (dimensions.m, dimensions.n, dimensions.k, iterations)
    if actual != defaults:
        command = [
            "env",
            f"RVAI_GEMM_M={dimensions.m}",
            f"RVAI_GEMM_N={dimensions.n}",
            f"RVAI_GEMM_K={dimensions.k}",
            f"RVAI_GEMM_ITERATIONS={iterations}",
            *command,
        ]
    return command


@app.command()
def report(
    result_path: Path = typer.Argument(..., help="Saved RunRecord JSON file."),
    report_format: ReportFormat = typer.Option(
        ReportFormat.MARKDOWN,
        "--format",
        help="Report output format.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write the report to a file instead of stdout.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing --output file.",
    ),
) -> None:
    """Render a validated benchmark RunRecord."""

    if force and output is None:
        _fail("--force requires --output")
    try:
        record = load_run_record(result_path)
        if report_format is ReportFormat.MARKDOWN:
            rendered = render_markdown(record)
        else:  # pragma: no cover - Typer rejects unsupported enum values.
            _fail(f"Unsupported report format: {report_format}")
        if output is not None:
            save_markdown_report(rendered, output, force=force)
        else:
            typer.echo(rendered, nl=False)
    except (ReportRenderError, ResultStoreError) as exc:
        _fail(str(exc))


@app.command()
def compare(
    left_path: Path = typer.Argument(..., help="First saved RunRecord JSON file."),
    right_path: Path = typer.Argument(..., help="Second saved RunRecord JSON file."),
) -> None:
    """Compare two benchmark records without unsafe performance claims."""

    try:
        comparison = compare_run_records(
            load_run_record(left_path),
            load_run_record(right_path),
        )
    except ResultStoreError as exc:
        _fail(str(exc))
    typer.echo(comparison.model_dump_json(indent=2))


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
        artifact_resolver = ArtifactResolver()
        report = CompatibilityMatcher(
            model_file_exists=lambda candidate: artifact_resolver.status(
                candidate
            ).verified
        ).check(manifest, hardware)
    except (
        ArtifactCacheError,
        CompatibilityError,
        HardwareProbeError,
        RegistryError,
    ) as exc:
        _fail(str(exc))
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
